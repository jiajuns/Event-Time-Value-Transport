"""Versioned multi-axis video annotation contract. Never manufacture rewards."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs/event_schema_factorized_v2.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
HEADS = {name: tuple(values) for name, values in SCHEMA["heads"].items()}
CORE_HEADS = ("phase", "holding", "placement")
DATA_FORMAT = "eksf_factorized_video_events_v2"
CHECKPOINT_FORMAT = "eksf_yolo_video_cfc_observer_v2"


def read_observations(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty observation manifest")
    keys, groups, episodes, times = set(), {}, {}, {}
    for row in rows:
        uid, group, split = row["attempt_uid"], row["group_uid"], row["split"]
        query, timestamp = row["query_id"], row["elapsed_s"]
        if (not isinstance(uid, str) or not uid or not isinstance(group, str) or not group or
                split not in ("train", "validation", "test", "inference") or
                type(query) is not int or query < 0 or not np.isfinite(timestamp) or timestamp < 0):
            raise ValueError("invalid episode/group/split/query/timestamp")
        if (uid, query) in keys:
            raise ValueError("duplicate observation identity")
        keys.add((uid, query))
        if group in groups and groups[group] != split:
            raise ValueError("one source group cannot cross splits")
        if uid in episodes and episodes[uid] != (group, split):
            raise ValueError("one episode cannot cross groups/splits")
        groups[group], episodes[uid] = split, (group, split)
        times.setdefault(uid, []).append((query, timestamp))
    for samples in times.values():
        ordered = sorted(samples)
        if len(ordered) > 1 and np.any(np.diff([x[1] for x in ordered]) <= 0):
            raise ValueError("query order must have strictly increasing timestamps")
    return rows


def interval_labels(path: Path | None, observations: list[dict]) -> np.ndarray:
    """Blank/gaps => -1. Approved unknown => its explicit class, not masked.

    Overlapping intervals for the same head are rejected, even with equal
    labels. Different heads may be annotated independently and overlap.
    """
    labels = np.full((len(observations), len(HEADS)), -1, dtype=np.int64)
    if path is None:
        return labels
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"attempt_uid", "start_s", "end_s", "reviewed", "reviewer", *HEADS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("annotation CSV missing columns")
        intervals = list(reader)
    known = {row["attempt_uid"] for row in observations}
    ranges = {}
    for interval in intervals:
        uid = interval["attempt_uid"]
        if uid not in known:
            raise ValueError("annotation references an unknown episode")
        if interval["reviewed"].strip().lower() not in ("true", "false"):
            raise ValueError("reviewed must be true or false")
        if interval["reviewed"].strip().lower() == "false":
            continue
        if not interval["reviewer"].strip():
            raise ValueError("reviewed annotations require reviewer identity")
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError("invalid annotation interval")
        for column, (head, classes) in enumerate(HEADS.items()):
            value = interval[head].strip()
            if not value:
                continue
            if value not in classes:
                raise ValueError(f"unknown {head} label: {value}")
            occupied = ranges.setdefault((uid, head), [])
            if any(max(start, a) < min(end, b) for a, b in occupied):
                raise ValueError("overlapping annotation intervals for same head")
            occupied.append((start, end))
            selected = [i for i, row in enumerate(observations)
                        if row["attempt_uid"] == uid and start <= row["elapsed_s"] < end]
            if not selected:
                raise ValueError("reviewed interval contains no sampled frame; increase sample rate or fix bounds")
            labels[selected, column] = classes.index(value)
    holding = list(HEADS).index("holding")
    placement = list(HEADS).index("placement")
    if np.any((labels[:, holding] == HEADS["holding"].index("held")) &
              (labels[:, placement] == HEADS["placement"].index("placed"))):
        raise ValueError("held and placed cannot both be true")
    return labels


def write_annotation_template(path: Path, observations: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["attempt_uid", "start_s", "end_s", *HEADS, "reviewed", "reviewer"])
        writer.writeheader()
        for uid in dict.fromkeys(row["attempt_uid"] for row in observations):
            times = sorted(row["elapsed_s"] for row in observations if row["attempt_uid"] == uid)
            end = times[-1] + (times[-1] - times[-2] if len(times) > 1 else .001)
            writer.writerow(dict(attempt_uid=uid, start_s=times[0], end_s=end, reviewed="false"))
