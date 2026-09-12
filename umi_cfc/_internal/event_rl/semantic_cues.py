"""Observation-only YOLOE cues. Detections are NOT event labels or rewards.

Slots describe task roles, not robot joints or detector-specific class indices.
Multiple candidates are explicitly ambiguous; this v1 never guesses identity.
Temporal reasoning stays in the CfC, not a single-frame overlap heuristic.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

FORMAT = "eksf_yoloe_semantic_cues_v1"
ROLES = ("object", "target", "gripper")
SLOT_FIELDS = ("present", "ambiguous", "count", "confidence", "cx", "cy", "width", "height")
PAIR_FIELDS = ("valid", "dx", "dy", "distance", "box_iou")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prompts(prompts: Mapping[str, Sequence[str]]) -> tuple[list[str], list[str]]:
    if set(prompts) != set(ROLES):
        raise ValueError(f"prompt roles must be exactly {ROLES}")
    names, roles = [], []
    for role in ROLES:
        values = prompts[role]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("each role needs a nonempty list of text prompts")
        for name in values:
            if not isinstance(name, str) or not name.strip() or name != name.strip() or name in names:
                raise ValueError("prompts must be nonempty, trimmed and unique across roles")
            names.append(name)
            roles.append(role)
    return names, roles


def feature_names(cameras: Sequence[str]) -> list[str]:
    if not cameras or len(set(cameras)) != len(cameras) or any(not c for c in cameras):
        raise ValueError("cameras must be nonempty and unique")
    return [f"{camera}.{group}.{field}" for camera in cameras for group, fields in
            [(role, SLOT_FIELDS) for role in ROLES] +
            [("object_to_gripper", PAIR_FIELDS), ("object_to_target", PAIR_FIELDS)]
            for field in fields]


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.prod(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection
    return intersection / max(union, 1e-12)


def frame_cues(detections: Sequence[Mapping], *, confidence: float = 0.25) -> np.ndarray:
    """Each detection: role, confidence, normalized xyxy (not pixel coordinates).

    Box geometry is emitted only for a unique candidate. Missing/ambiguous
    geometry is zero with an explicit validity signal; zero is NOT absence of
    physical contact, a drop label, or task failure. No cross-camera geometry.
    """
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence threshold must be in [0,1]")
    grouped = {role: [] for role in ROLES}
    for detection in detections:
        role = detection["role"]
        score = float(detection["confidence"])
        box = np.asarray(detection["xyxy"], dtype=np.float64)
        if role not in grouped or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid detection role/confidence")
        if (box.shape != (4,) or not np.isfinite(box).all() or
                np.any(box < 0) or np.any(box > 1) or np.any(box[2:] <= box[:2])):
            raise ValueError("boxes must be finite, nondegenerate normalized xyxy")
        if score >= confidence:
            grouped[role].append((score, box))
    values, selected = [], {}
    for role in ROLES:
        candidates = grouped[role]
        count = len(candidates)
        slot = [float(count > 0), float(count > 1), float(count),
                max((x[0] for x in candidates), default=0.0), 0., 0., 0., 0.]
        if count == 1:
            box = candidates[0][1]
            selected[role] = box
            slot[4:] = [*((box[:2] + box[2:]) / 2), *(box[2:] - box[:2])]
        values.extend(slot)
    for other in ("gripper", "target"):
        if "object" not in selected or other not in selected:
            values.extend([0.] * len(PAIR_FIELDS))
            continue
        obj, dest = selected["object"], selected[other]
        delta = (dest[:2] + dest[2:] - obj[:2] - obj[2:]) / 2
        values.extend([1., *delta, float(np.linalg.norm(delta)), box_iou(obj, dest)])
    return np.asarray(values, dtype=np.float32)


class YOLOECueDetector:
    """Lazy optional dependency; text prompts are set once, not per image.

    weights must already be local. Ultralytics may fetch its text encoder on
    first set_classes(); provision/cache it before an offline deployment.
    """
    def __init__(self, weights: Path, prompts: Mapping[str, Sequence[str]], *,
                 device: str = "cpu", imgsz: int = 640, confidence: float = .25):
        if not weights.is_file():
            raise ValueError("supply an existing local YOLOE text-prompt checkpoint")
        names, self.roles = validate_prompts(prompts)
        if imgsz <= 0 or not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid image size/confidence threshold")
        from ultralytics import YOLOE
        from importlib.metadata import version
        self.model = YOLOE(str(weights))
        self.model.set_classes(names)
        self.options = dict(device=device, imgsz=imgsz, conf=confidence, verbose=False)
        self.provenance = dict(backend="ultralytics.YOLOE", version=version("ultralytics"),
                               weights_sha256=sha256(weights), weights_name=weights.name,
                               prompts=dict(prompts), **self.options)

    def __call__(self, image: Path) -> list[dict]:
        import torch
        with torch.inference_mode():
            result = self.model.predict(source=str(image), **self.options)[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [dict(role=self.roles[int(cls)], confidence=float(score), xyxy=box.tolist())
                for box, score, cls in zip(boxes.xyxyn.cpu().numpy(),
                                           boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]


def append_semantic_cues(arrays: Mapping[str, np.ndarray], path: Path) -> tuple[dict, dict]:
    """Exact keyed join, causal timestamp check, preserve source labels/actions.

    This does not infer events. The current CfC consumes the appended cues as
    vision features and is still supervised by the existing reviewed labels.
    """
    with np.load(path, allow_pickle=False) as archive:
        contract = json.loads(str(archive["contract_json"].item()))
        uid = archive["attempt_uid"].astype(str)
        query = archive["query_id"]
        times = archive["elapsed_s"]
        cues = archive["semantic_features"]
    if contract.get("format") != FORMAT or contract.get("event_labels_generated") is not False:
        raise ValueError("not an observation-only semantic cue sidecar")
    names = feature_names(contract["cameras"])
    if contract.get("feature_names") != names:
        raise ValueError("semantic feature schema/order mismatch")
    n = len(uid)
    if (uid.shape != (n,) or query.shape != (n,) or times.shape != (n,) or
            query.dtype.kind not in "iu" or cues.shape != (n, len(names)) or
            not np.isfinite(cues).all() or not np.isfinite(times).all()):
        raise ValueError("invalid semantic sidecar shapes/values")
    keys = list(zip(uid.tolist(), query.tolist()))
    expected = list(zip(np.asarray(arrays["attempt_uid"]).astype(str).tolist(),
                        np.asarray(arrays["query_id"]).tolist()))
    if len(set(keys)) != n or len(set(expected)) != len(expected) or set(keys) != set(expected):
        raise ValueError("semantic keys must match source queries exactly, without duplicates")
    lookup = {key: index for index, key in enumerate(keys)}
    order = np.asarray([lookup[key] for key in expected], dtype=np.int64)
    if not np.allclose(times[order], arrays["elapsed_s"], rtol=0, atol=1e-5):
        raise ValueError("semantic timestamps must match current query observations")
    original = np.asarray(arrays["vision_features"], dtype=np.float32)
    combined = dict(arrays)
    combined["vision_features"] = np.concatenate((original, cues[order]), axis=1).astype(np.float32)
    return combined, dict(contract, sidecar_sha256=sha256(path),
                          base_vision_dim=original.shape[1], semantic_dim=len(names),
                          fusion="append_to_vision_features_before_train_only_normalization")
