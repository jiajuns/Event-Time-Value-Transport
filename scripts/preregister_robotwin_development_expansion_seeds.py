#!/usr/bin/env python3
"""Resolve 150 development-only RoboTwin scenes without policy/label access."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from robotwin_development_seed_contract import (
    CANDIDATE_FORMAT,
    CANDIDATE_STATUS,
    EXPECTED_COUNT,
    LABEL_ACCESS_CONTRACT,
    MANIFEST_FORMAT,
    MANIFEST_STATUS,
    PURPOSE,
    REGISTRY,
    canonical_sha256,
    expand_candidate_requested_seeds,
    fresh_seed_sets,
    official_seeds,
    sha256,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_reset_unique_scenes(
    candidates: list[int],
    *,
    resolver: Callable[[int], int],
    excluded: set[int],
    count: int = EXPECTED_COUNT,
) -> tuple[list[dict[str, int]], list[dict[str, int | str]]]:
    selected: list[dict[str, int]] = []
    resolved_seen = set(excluded)
    audit: list[dict[str, int | str]] = []
    for requested in candidates:
        resolved = int(resolver(requested))
        if requested in excluded:
            decision = "requested_seed_in_official_or_fresh_exclusion"
        elif resolved in excluded:
            decision = "resolved_scene_in_official_or_fresh_exclusion"
        elif resolved in resolved_seen:
            decision = "duplicate_resolved_development_scene"
        else:
            decision = "selected"
            resolved_seen.add(resolved)
            selected.append(
                {
                    "seed": requested,
                    "requested_seed": requested,
                    "resolved_seed": resolved,
                }
            )
        audit.append(
            {
                "requested_seed": requested,
                "resolved_seed": resolved,
                "decision": decision,
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only {len(selected)} reset-unique development scenes resolved; need {count}"
        )
    return selected, audit


def build_manifest(
    *,
    task: str,
    selected: list[dict[str, int]],
    audit: list[dict[str, int | str]],
    candidate_path: Path,
    official_path: Path,
    fresh_path: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "format": MANIFEST_FORMAT,
        "schema_version": 1,
        "status": MANIFEST_STATUS,
        "task": task,
        "purpose": PURPOSE,
        "seed_registry": REGISTRY,
        "train": selected,
        "requested_seeds": [row["requested_seed"] for row in selected],
        "resolved_seeds": [row["resolved_seed"] for row in selected],
        "candidate_manifest": str(candidate_path.resolve()),
        "candidate_manifest_sha256": sha256(candidate_path),
        "candidate_contract": {
            "format": CANDIDATE_FORMAT,
            "selection_rule": candidate.get("selection_rule"),
            "freeze_rule": candidate.get("freeze_rule"),
            "candidate_seed_range": candidate.get("candidate_seed_range"),
            "candidate_count": len(expand_candidate_requested_seeds(candidate)),
        },
        "official_seed_registry": str(official_path.resolve()),
        "official_seed_registry_sha256": sha256(official_path),
        "fresh_seed_manifest": str(fresh_path.resolve()),
        "fresh_seed_manifest_sha256": sha256(fresh_path),
        "selection_rule": candidate.get("selection_rule"),
        "freeze_rule": candidate.get("freeze_rule"),
        "audit": audit,
        "label_access_contract": LABEL_ACCESS_CONTRACT,
        "prohibited_use": (
            "never_use_as_fresh_confirmation_or_select_by_policy_success_event_reward"
        ),
    }
    result["manifest_payload_sha256"] = canonical_sha256(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--fresh-seed-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--official-seed-registry", type=Path)
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--count", type=int, default=EXPECTED_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen development manifest: {args.output}"
        )
    if args.count != EXPECTED_COUNT:
        raise ValueError(f"formal development expansion requires count={EXPECTED_COUNT}")
    candidate = json.loads(args.candidates.read_text(encoding="utf-8"))
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("format") != CANDIDATE_FORMAT
        or candidate.get("status") != CANDIDATE_STATUS
        or str(candidate.get("task", "")) != args.task
        or candidate.get("purpose") != PURPOSE
    ):
        raise RuntimeError("candidate seed manifest is not frozen development input")
    candidates = expand_candidate_requested_seeds(candidate)
    official_path = (
        args.official_seed_registry
        if args.official_seed_registry is not None
        else args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    ).resolve()
    fresh_path = args.fresh_seed_manifest.resolve()
    official = json.loads(official_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    official_values = official_seeds(official, args.task)
    fresh_requested, fresh_resolved = fresh_seed_sets(fresh, args.task)
    excluded = set(official_values) | set(fresh_requested) | set(fresh_resolved)

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    from collect_openvla_etsf_candidate_branches import reset_with_contract
    from collect_openvla_etsf_rollouts import environment_config

    env = RoboTwinEnv(
        cfg=environment_config(
            args.robotwin_root, official_path, args.task, len(candidates)
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )
    try:
        def resolve(requested: int) -> int:
            # Identity/language only: no policy, action, reward, event or success.
            _, _, resolved, _ = reset_with_contract(env, requested)
            return int(resolved)

        selected, audit = select_reset_unique_scenes(
            candidates, resolver=resolve, excluded=excluded, count=args.count
        )
    finally:
        env.venv.close(clear_cache=False)
    result = build_manifest(
        task=args.task,
        selected=selected,
        audit=audit,
        candidate_path=args.candidates,
        official_path=official_path,
        fresh_path=fresh_path,
        candidate=candidate,
    )
    atomic_json(args.output, result)
    print(
        "DEVELOPMENT_EXPANSION_PREREGISTERED="
        + json.dumps(
            {
                "output": str(args.output.resolve()),
                "count": len(selected),
                "manifest_file_sha256": sha256(args.output),
                "manifest_payload_sha256": result["manifest_payload_sha256"],
                "labels_read": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
