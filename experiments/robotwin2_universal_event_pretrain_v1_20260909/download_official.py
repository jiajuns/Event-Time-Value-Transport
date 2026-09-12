#!/usr/bin/env python3
"""Download and verify the immutable RoboTwin clean expert replay slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from universal_event.schema import TASK_SPECS


FORMAT = "etsf_robotwin2_universal_official_replay_slice_v1"
REPO_ID = "TianxingChen/RoboTwin2.0"
REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
UNAVAILABLE_PAIRS = {
    ("place_can_basket", "piper"),
    ("put_object_cabinet", "piper"),
    ("put_object_cabinet", "ur5"),
    ("hanging_mug", "piper"),
    ("handover_block", "franka"),
    ("handover_block", "ur5"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, minimum_episodes: int) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {path}: {bad}")
        seed_names = [name for name in archive.namelist() if name.endswith("/seed.txt")]
        if len(seed_names) != 1:
            raise ValueError(f"invalid seed.txt count in {path}")
        root = seed_names[0].rsplit("/", 1)[0]
        seeds = [int(value) for value in archive.read(seed_names[0]).decode().split()]
        if len(seeds) < minimum_episodes or len(set(seeds)) != len(seeds):
            raise ValueError(f"insufficient or repeated seeds in {path}")
        missing = [index for index in range(len(seeds))
                   if f"{root}/_traj_data/episode{index}.pkl" not in archive.namelist()]
        if missing:
            raise ValueError(f"missing {len(missing)} replay paths in {path}")
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path),
        "episodes": len(seeds), "first_seed": seeds[0], "last_seed": seeds[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=50)
    parser.add_argument("--tasks", nargs="*", choices=tuple(TASK_SPECS),
                        default=list(TASK_SPECS))
    parser.add_argument("--receipt-name", default="universal_event_replay_receipt.json")
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    selected = list(dict.fromkeys(args.tasks))
    expected = [(task, body) for task in selected for body in BODIES
                if (task, body) not in UNAVAILABLE_PAIRS]
    for task, body in expected:
        filename = f"dataset/{task}/{body}_clean_50.zip"
        try:
            downloaded = Path(hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                filename=filename, local_dir=root,
            ))
            row = validate_archive(downloaded, args.minimum_episodes)
            row.update(task=task, body=body, repo_path=filename)
            rows.append(row)
            print(json.dumps({"verified": filename, "bytes": row["bytes"]}), flush=True)
        except Exception as exc:
            errors.append({"task": task, "body": body, "repo_path": filename,
                           "error": f"{type(exc).__name__}: {exc}"})
            print(json.dumps(errors[-1]), flush=True)
    receipt = {
        "format": FORMAT, "repo_id": REPO_ID, "revision": REVISION,
        "tasks": selected, "bodies": list(BODIES),
        "officially_unavailable_pairs": [list(pair) for pair in sorted(UNAVAILABLE_PAIRS)
                                          if pair[0] in selected],
        "expected_archives": len(expected),
        "verified_archives": len(rows), "files": rows, "errors": errors,
    }
    receipt_path = root / args.receipt_name
    if receipt_path.parent != root or receipt_path.name != args.receipt_name:
        raise ValueError("receipt-name must be a plain filename")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    if errors or len(rows) != len(expected):
        raise RuntimeError(f"official replay slice incomplete: {len(rows)} verified, {len(errors)} errors")


if __name__ == "__main__":
    main()
