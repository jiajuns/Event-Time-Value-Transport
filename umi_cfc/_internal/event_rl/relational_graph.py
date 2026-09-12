"""Causal per-observation object graphs built from recorded YOLO boxes."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

GRAPH_VERSION = "umi_relational_graph_v3"
ROLES = ("object", "target", "gripper")
NODE_FEATURES = (
    "role_object", "role_target", "role_gripper", "x1", "y1", "x2", "y2",
    "confidence", "track_age_normalized", "delta_x1", "delta_y1", "delta_x2",
    "delta_y2", "missing", "ambiguity",
)
EDGE_FEATURES = (
    "center_dx", "center_dy", "log_width_ratio", "log_height_ratio", "iou",
    "center_distance", "valid",
)


def box_iou(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    aa = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    bb = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    role: int
    box: np.ndarray
    age: int = 1
    missing: int = 0


class CausalGraphTracker:
    """Greedy same-role IoU tracker; state contains past/current observations only."""

    def __init__(self, max_nodes=8, max_missing=2, iou_threshold=.2, ambiguity_iou=.05):
        if max_nodes <= 0 or max_missing < 0 or not 0 <= iou_threshold <= 1:
            raise ValueError("invalid tracker configuration")
        self.max_nodes, self.max_missing = max_nodes, max_missing
        self.iou_threshold, self.ambiguity_iou = iou_threshold, ambiguity_iou
        self.tracks = [None] * max_nodes
        self.next_track_id = 0

    def update(self, detections):
        clean = []
        for detection in detections:
            role = detection.get("role")
            box = np.asarray(detection.get("xyxy"), dtype=np.float32)
            confidence = float(detection.get("confidence", np.nan))
            if role not in ROLES or box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(confidence):
                raise ValueError("invalid detection")
            if np.any(box < 0) or np.any(box > 1) or box[2] <= box[0] or box[3] <= box[1] or not 0 <= confidence <= 1:
                raise ValueError("invalid normalized detection box/confidence")
            clean.append(dict(role=ROLES.index(role), box=box, confidence=confidence))
        raw_role_counts = np.bincount([item["role"] for item in clean], minlength=len(ROLES))
        self.last_truncated = len(clean) > self.max_nodes
        # Keep multiple candidates; only capacity overflow is resolved by confidence.
        clean = sorted(clean, key=lambda x: x["confidence"], reverse=True)[:self.max_nodes]
        unmatched_tracks = {i for i, track in enumerate(self.tracks) if track is not None}
        unmatched_detections = set(range(len(clean)))
        matches = []
        candidates = sorted(
            ((box_iou(track.box, detection["box"]), slot, di)
             for slot, track in enumerate(self.tracks) if track is not None
             for di, detection in enumerate(clean) if track.role == detection["role"]),
            reverse=True,
        )
        for overlap, slot, di in candidates:
            if overlap < self.iou_threshold:
                break
            if slot in unmatched_tracks and di in unmatched_detections:
                matches.append((slot, di)); unmatched_tracks.remove(slot); unmatched_detections.remove(di)
        previous_boxes = {}
        for slot, di in matches:
            track = self.tracks[slot]; previous_boxes[slot] = track.box.copy()
            track.box = clean[di]["box"]; track.age += 1; track.missing = 0
        for slot in unmatched_tracks:
            track = self.tracks[slot]; track.age += 1; track.missing += 1
            if track.missing > self.max_missing:
                self.tracks[slot] = None
        free = [i for i, track in enumerate(self.tracks) if track is None]
        needed = len(unmatched_detections) - len(free)
        if needed > 0:
            # A current observation outranks stale missing state. Eviction never reuses its ID.
            stale = sorted((i for i, track in enumerate(self.tracks)
                            if track is not None and track.missing > 0),
                           key=lambda i: self.tracks[i].missing, reverse=True)
            for slot in stale[:needed]:
                self.tracks[slot] = None
                free.append(slot)
        for di in sorted(unmatched_detections, key=lambda i: clean[i]["confidence"], reverse=True):
            if not free:
                break
            slot = free.pop(0)
            self.tracks[slot] = _Track(self.next_track_id, clean[di]["role"], clean[di]["box"])
            self.next_track_id += 1
            matches.append((slot, di)); previous_boxes[slot] = clean[di]["box"].copy()

        current = {slot: clean[di] for slot, di in matches}
        role_counts = np.bincount([item["role"] for item in current.values()], minlength=len(ROLES))
        features = np.zeros((self.max_nodes, len(NODE_FEATURES)), np.float32)
        boxes = np.zeros((self.max_nodes, 4), np.float32)
        mask = np.zeros(self.max_nodes, bool)
        track_ids = np.full(self.max_nodes, -1, np.int64)
        for slot, track in enumerate(self.tracks):
            if track is None:
                continue
            track_ids[slot] = track.track_id
            features[slot, track.role] = 1
            features[slot, 8] = min(track.age / 100.0, 1.0)
            if slot not in current:
                features[slot, 13] = 1
                continue
            item = current[slot]; box = item["box"]
            mask[slot] = True; boxes[slot] = box
            features[slot, 3:7] = box
            features[slot, 7] = item["confidence"]
            features[slot, 9:13] = box - previous_boxes[slot]
            alternatives = sum(1 for other in current.values()
                               if other is not item and other["role"] == item["role"]
                               and box_iou(other["box"], box) >= self.ambiguity_iou)
            features[slot, 14] = float(raw_role_counts[item["role"]] > 1 or alternatives > 0)
        bindings = np.full(len(ROLES), -1, np.int64)
        for role in range(len(ROLES)):
            slots = [slot for slot, item in current.items() if item["role"] == role]
            if len(slots) == 1 and raw_role_counts[role] == 1:
                bindings[role] = slots[0]
        return features, mask, boxes, track_ids, edge_features(boxes, mask), bindings


def edge_features(boxes, mask):
    n = len(boxes); result = np.zeros((n, n, len(EDGE_FEATURES)), np.float32)
    for source in np.flatnonzero(mask):
        a = boxes[source]; ac = (a[:2] + a[2:]) / 2; awh = np.maximum(a[2:] - a[:2], 1e-6)
        for target in np.flatnonzero(mask):
            b = boxes[target]; bc = (b[:2] + b[2:]) / 2; bwh = np.maximum(b[2:] - b[:2], 1e-6)
            delta = bc - ac
            result[source, target] = (*delta, *np.log(bwh / awh), box_iou(a, b),
                                      float(np.linalg.norm(delta)), 1.0)
    return result
