#!/usr/bin/env python3
"""Resolve a pre-registered RoboTwin seed pool without running any policy.

This utility is intentionally reset-only.  It never loads a VLA, executes an
action, derives an event, or reads success.  The first ``count`` stable,
resolved-unique scenes are frozen for a later confirmation collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen confirmation manifest: {args.output}"
        )
    if args.count <= 0:
        raise ValueError("--count must be positive")
    proposal = json.loads(args.candidates.read_text(encoding="utf-8"))
    if proposal.get("status") != "preregistered_unresolved":
        raise RuntimeError("candidate seed manifest is not frozen unresolved input")
    if str(proposal.get("task")) != args.task:
        raise RuntimeError("candidate seed task differs from --task")
    candidates = [int(value) for value in proposal["candidate_requested_seeds"]]
    if len(candidates) < args.count or len(set(candidates)) != len(candidates):
        raise RuntimeError("candidate pool is too small or contains duplicates")

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    from collect_openvla_etsf_candidate_branches import reset_with_contract
    from collect_openvla_etsf_rollouts import environment_config, load_official_seeds

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    official = set(load_official_seeds(seeds_path, args.task, 150, 0))
    overlap = sorted(set(candidates) & official)
    if overlap:
        raise RuntimeError(f"fresh candidate pool overlaps official seeds: {overlap}")

    env = RoboTwinEnv(
        cfg=environment_config(
            args.robotwin_root, seeds_path, args.task, len(candidates)
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    selected: list[dict[str, int]] = []
    resolved_seen = set(official)
    audited: list[dict[str, int | str]] = []
    try:
        for requested in candidates:
            # reset_with_contract returns identity/language only.  Do not touch
            # reward, success, task-state, event, image, or action interfaces.
            _, _, resolved, _ = reset_with_contract(env, requested)
            reason = "selected"
            if resolved in resolved_seen:
                reason = "duplicate_or_official_resolved_scene"
            else:
                resolved_seen.add(resolved)
                selected.append(
                    {"seed": requested, "requested_seed": requested, "resolved_seed": resolved}
                )
            audited.append(
                {
                    "requested_seed": requested,
                    "resolved_seed": resolved,
                    "decision": reason,
                }
            )
            if len(selected) == args.count:
                break
    finally:
        env.venv.close(clear_cache=False)
    if len(selected) != args.count:
        raise RuntimeError(
            f"only {len(selected)} unique fresh scenes resolved; need {args.count}"
        )

    result = {
        "schema_version": 1,
        "status": "fresh_confirmation_preregistered_resolved",
        "task": args.task,
        "test": selected,
        "requested_seeds": [row["requested_seed"] for row in selected],
        "resolved_seeds": [row["resolved_seed"] for row in selected],
        "candidate_manifest": str(args.candidates.resolve()),
        "candidate_manifest_sha256": sha256(args.candidates),
        "official_seed_registry": str(seeds_path.resolve()),
        "official_seed_registry_sha256": sha256(seeds_path),
        "selection_rule": proposal["selection_rule"],
        "freeze_rule": proposal["freeze_rule"],
        "audit": audited,
        "label_access_contract": (
            "reset_identity_only_no_policy_no_action_no_event_no_success_no_reward"
        ),
    }
    atomic_json(args.output, result)
    print(
        "FRESH_CONFIRMATION_PREREGISTERED="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": len(selected),
                "candidate_manifest_sha256": result["candidate_manifest_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
