"""Causal pairwise visual context for relational observer v4."""
from __future__ import annotations

import cv2
import numpy as np

VISUAL_VERSION = "umi_relational_visual_v4"
PAIR_NAMES = ("object_gripper", "object_target")
PAIR_FEATURE_NAMES = (
    "relative_center_dx", "relative_center_dy", "log_width_ratio", "log_height_ratio",
    "iou", "center_distance", "role_a_visible", "role_b_visible",
    "role_a_ambiguous", "role_b_ambiguous", "relative_delta_dx",
    "relative_delta_dy", "role_a_history_position", "role_b_history_position",
    "full_frame_fallback", "current_geometry_valid",
)
PAIR_ROLES = ((0, 2), (0, 1))  # object-gripper, object-target


def _iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    areas = [float(np.prod(np.maximum(box[2:] - box[:2], 0))) for box in (a, b)]
    union = areas[0] + areas[1] - intersection
    return intersection / union if union > 0 else 0.0


def relative_geometry(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ac, bc = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
    awh = np.maximum(a[2:] - a[:2], 1e-6); bwh = np.maximum(b[2:] - b[:2], 1e-6)
    delta = bc - ac
    return np.asarray((*delta, *np.log(bwh / awh), _iou(a, b), np.linalg.norm(delta)), np.float32)


class CausalPairVisualState:
    """Use unique current boxes or recent past boxes; never inspect future rows."""

    def __init__(self, max_history_s=.3):
        if not np.isfinite(max_history_s) or max_history_s < 0:
            raise ValueError("invalid history horizon")
        self.max_history_s = max_history_s
        self.last = {}
        self.previous_relative = [None] * len(PAIR_ROLES)
        self.time = None

    def update(self, elapsed_s, node_features, node_mask, node_boxes, bindings):
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or (self.time is not None and elapsed_s <= self.time):
            raise ValueError("timestamps must be finite and strictly increasing")
        if self.time is not None and elapsed_s - self.time > .151:
            self.last.clear(); self.previous_relative = [None] * len(PAIR_ROLES)
        self.time = elapsed_s
        node_features = np.asarray(node_features); node_mask = np.asarray(node_mask, bool)
        node_boxes = np.asarray(node_boxes); bindings = np.asarray(bindings, np.int64)
        role_counts = np.asarray([np.count_nonzero(node_mask & (node_features[:, role] > .5))
                                  for role in range(3)])
        visible = role_counts > 0
        ambiguous = role_counts > 1
        current = {}
        for role in range(3):
            slot = int(bindings[role])
            if role_counts[role] == 1 and 0 <= slot < len(node_mask) and node_mask[slot]:
                current[role] = node_boxes[slot].astype(np.float32, copy=True)
                self.last[role] = (elapsed_s, current[role])
        boxes, features, crop_boxes = {}, [], []
        for pair_index, (role_a, role_b) in enumerate(PAIR_ROLES):
            chosen, historical = [], []
            for role in (role_a, role_b):
                if role in current:
                    chosen.append(current[role]); historical.append(False)
                elif role in self.last and elapsed_s - self.last[role][0] <= self.max_history_s + 1e-9:
                    chosen.append(self.last[role][1]); historical.append(True)
                else:
                    chosen.append(None); historical.append(False)
            current_valid = role_a in current and role_b in current
            geometry = relative_geometry(*chosen) if all(box is not None for box in chosen) else np.zeros(6, np.float32)
            prior = self.previous_relative[pair_index]
            relative_delta = geometry[:2] - prior if prior is not None and all(box is not None for box in chosen) else np.zeros(2, np.float32)
            if all(box is not None for box in chosen):
                self.previous_relative[pair_index] = geometry[:2].copy()
            feature = np.asarray((*geometry, visible[role_a], visible[role_b],
                                  ambiguous[role_a], ambiguous[role_b], *relative_delta,
                                  historical[0], historical[1], not current_valid, current_valid), np.float32)
            features.append(feature)
            crop_boxes.append(chosen if current_valid else None)
        return np.stack(features), crop_boxes


def crop_pair_images(frame_bgr, crop_boxes, size=96, margin=.15):
    """Crop union regions with context; invalid/ambiguous pairs fall back to full frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.dtype != np.uint8:
        raise ValueError("frame must be BGR uint8 HWC")
    height, width = frame_bgr.shape[:2]
    result = []
    for pair in crop_boxes:
        if pair is None:
            roi = frame_bgr
        else:
            union = np.asarray((np.minimum(pair[0][:2], pair[1][:2]),
                                np.maximum(pair[0][2:], pair[1][2:])), np.float32).reshape(4)
            extent = np.maximum(union[2:] - union[:2], .05)
            union[:2] -= margin * extent; union[2:] += margin * extent
            union = np.clip(union, 0, 1)
            left, top = int(np.floor(union[0] * width)), int(np.floor(union[1] * height))
            right, bottom = int(np.ceil(union[2] * width)), int(np.ceil(union[3] * height))
            roi = frame_bgr[top:bottom, left:right] if right > left and bottom > top else frame_bgr
        resized = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
        result.append(np.moveaxis(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), -1, 0))
    return np.stack(result).astype(np.uint8, copy=False)
