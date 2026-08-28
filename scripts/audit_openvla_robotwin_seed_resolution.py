#!/usr/bin/env python3
"""Audit resolved RoboTwin Piper seeds before OpenVLA candidate splits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from collect_openvla_etsf_candidate_branches import reset_with_contract
from collect_openvla_etsf_rollouts import (
    BODY,
    DEFAULT_TASK,
    environment_config,
    load_official_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()

    # RLinf's RoboTwin adapter selects the Piper-compatible task wrapper through
    # the same platform contract used by the formal OpenVLA collector.
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    seeds = load_official_seeds(seeds_path, args.task, args.limit, args.offset)
    env = RoboTwinEnv(
        cfg=environment_config(args.robotwin_root, seeds_path, args.task, len(seeds)),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    rows = []
    try:
        for index, seed in enumerate(seeds, start=args.offset):
            _, _, resolved, _ = reset_with_contract(env, seed)
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
    grouped: dict[int, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["resolved_seed"], []).append(row["requested_seed"])
    payload = {
        "task": args.task,
        "body": BODY,
        "offset": args.offset,
        "limit": args.limit,
        "rows": rows,
        "retried": [row for row in rows if row["retried"]],
        "duplicate_resolved_scenes": {
            str(seed): requested
            for seed, requested in grouped.items()
            if len(requested) > 1
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
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
