#!/usr/bin/env python3
"""Resolve RoboTwin official seeds after unstable-scene automatic retries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from collect_smolvla_etsf_candidate_branches import (
    BODY,
    DEFAULT_TASK,
    environment_config,
    load_official_seeds,
    reset_with_resolved_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    requested = load_official_seeds(
        seeds_path, args.task, args.limit, args.offset
    )
    env = RoboTwinEnv(
        cfg=environment_config(
            args.robotwin_root, seeds_path, args.task, len(requested), 200
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    rows = []
    try:
        for index, seed in enumerate(requested, start=args.offset):
            _, _, resolved, _ = reset_with_resolved_seed(env, seed)
            row = {
                "official_index": index,
                "requested_seed": seed,
                "resolved_seed": resolved,
                "retried": resolved != seed,
            }
            rows.append(row)
            print("RESOLVED=" + json.dumps(row, sort_keys=True), flush=True)
    finally:
        env.venv.close(clear_cache=False)

    by_resolved: dict[int, list[int]] = {}
    for row in rows:
        by_resolved.setdefault(row["resolved_seed"], []).append(
            row["requested_seed"]
        )
    payload = {
        "task": args.task,
        "body": BODY,
        "offset": args.offset,
        "limit": args.limit,
        "rows": rows,
        "retried": [row for row in rows if row["retried"]],
        "duplicate_resolved_scenes": {
            str(seed): values
            for seed, values in by_resolved.items()
            if len(values) > 1
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "AUDIT_COMPLETE="
        + json.dumps(
            {
                "rows": len(rows),
                "retried": len(payload["retried"]),
                "duplicate_resolved_scenes": payload[
                    "duplicate_resolved_scenes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
