#!/usr/bin/env python3
"""Download and safely extract official RoboTwin clean source trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import tempfile
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import hf_hub_download


TASKS = [
    "adjust_bottle",
    "handover_block",
    "move_can_pot",
    "place_container_plate",
    "beat_block_hammer",
    "lift_pot",
]
BODIES = {
    "aloha-agilex": "aloha-agilex_clean_50.zip",
    "ARX-X5": "arx-x5_clean_50.zip",
}


def validate_archive(archive: Path) -> None:
    with ZipFile(archive) as handle:
        members = [item for item in handle.infolist() if not item.is_dir()]
        if not members:
            raise ValueError(f"empty archive: {archive}")
        for item in members:
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe member in {archive}: {item.filename}")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"symlink member in {archive}: {item.filename}")


def inspect_payload(payload: Path) -> dict[str, object]:
    seeds = [int(x) for x in (payload / "seed.txt").read_text().split()]
    trajectories = sorted((payload / "_traj_data").glob("episode*.pkl"))
    data = sorted((payload / "data").glob("episode*.hdf5"))
    if len(seeds) != 50 or len(trajectories) != 50 or len(data) != 50:
        raise ValueError(
            f"expected 50 seeds/pickles/hdf5 in {payload}; got "
            f"{len(seeds)}/{len(trajectories)}/{len(data)}"
        )
    return {
        "seeds": len(seeds),
        "trajectory_pickles": len(trajectories),
        "hdf5_episodes": len(data),
        "videos": len(list((payload / "video").glob("episode*.mp4"))),
        "instructions": len(list((payload / "instructions").glob("episode*.json"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/user/etsf_stage1"))
    parser.add_argument("--repo", default="TianxingChen/RoboTwin2.0")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    archive_root = args.root / "source_archives"
    data_root = args.root / "source_data"
    archive_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for task in TASKS:
        for body, filename in BODIES.items():
            target = data_root / task / body
            if target.exists():
                counts = inspect_payload(target)
                print(f"SOURCE_READY task={task} body={body} (existing)", flush=True)
            else:
                remote_name = f"dataset/{task}/{filename}"
                archive = Path(
                    hf_hub_download(
                        repo_id=args.repo,
                        repo_type="dataset",
                        revision=args.revision,
                        filename=remote_name,
                        local_dir=archive_root,
                    )
                )
                validate_archive(archive)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=data_root) as tmp_name:
                    tmp = Path(tmp_name)
                    with ZipFile(archive) as handle:
                        handle.extractall(tmp)
                    roots = [item for item in tmp.iterdir() if item.is_dir()]
                    if len(roots) != 1:
                        raise ValueError(f"expected one payload root in {archive}: {roots}")
                    counts = inspect_payload(roots[0])
                    shutil.move(str(roots[0]), target)
                print(f"SOURCE_READY task={task} body={body}", flush=True)
            manifest.append(
                {
                    "task": task,
                    "embodiment": body,
                    "archive": filename,
                    "path": str(target),
                    **counts,
                }
            )

    (args.root / "source_manifest.json").write_text(
        json.dumps(
            {
                "repo": args.repo,
                "revision": args.revision,
                "tasks": TASKS,
                "embodiments": list(BODIES),
                "entries": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"WROTE {args.root / 'source_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
