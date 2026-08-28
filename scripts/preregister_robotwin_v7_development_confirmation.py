#!/usr/bin/env python3
"""Resolve v7 prospective development seeds using reset identity only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from openvla_etsf_v7_development_confirmation import (
    EXPECTED_GROUPS, TASK, canonical_sha256, expand_candidates, identity_sets,
    make_seed_manifest, select_reset_unique_scenes, sha256, validate_seed_manifest,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping): raise RuntimeError(f"expected JSON object: {path}")
    return value


def exclusion_source(path: Path, *, expected: int, name: str) -> tuple[dict[str, Any], set[int]]:
    path = path.expanduser().resolve()
    value = _json(path)
    identity_basis = "recorded_requested_and_resolved"
    if name == "official150" and not value.get("requested_seeds"):
        task_row = value.get(TASK)
        seeds = task_row.get("success_seeds") if isinstance(task_row, Mapping) else None
        if not isinstance(seeds, list) or len(seeds) != expected or len(set(map(int, seeds))) != expected:
            raise RuntimeError("official150 registry lacks 150 unique success seeds")
        requested = resolved = list(map(int, seeds))
        identity_basis = "official_registry_requested_equals_resolved_fallback"
    else:
        requested, resolved = identity_sets(value, expected=expected, name=name)
    return ({"path": str(path), "sha256": sha256(path), "requested_seeds": requested,
             "resolved_seeds": resolved,
             "identity_basis": identity_basis,
             "identity_sets_sha256": canonical_sha256({"requested": requested, "resolved": resolved})},
            set(requested) | set(resolved))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--official150-manifest", type=Path, required=True)
    parser.add_argument("--development150-manifest", type=Path, required=True)
    parser.add_argument("--fresh50-manifest", type=Path, required=True)
    parser.add_argument("--official-seed-registry", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--count", type=int, default=EXPECTED_GROUPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing v7 overwrite: {args.output}")
    if args.task != TASK or args.count != EXPECTED_GROUPS:
        raise ValueError("formal v7 is frozen to move_can_pot and 250 groups")
    candidate_path = args.candidates.resolve(); candidate = _json(candidate_path)
    candidates = expand_candidates(candidate)
    sources, excluded = {}, set()
    for key, path, count in (
        ("official150", args.official150_manifest, 150),
        ("development150", args.development150_manifest, 150),
        ("fresh50", args.fresh50_manifest, 50),
    ):
        source, identities = exclusion_source(path, expected=count, name=key)
        sources[key] = source; excluded.update(identities)
    official = args.official_seed_registry.resolve()
    if not official.is_file(): raise FileNotFoundError(official)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root.resolve())
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path[:0] = [str(args.rlinf_root.resolve()), str(args.robotwin_code.resolve()),
                    str(Path(__file__).resolve().parent)]
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from collect_openvla_etsf_candidate_branches import reset_with_contract
    from collect_openvla_etsf_rollouts import environment_config
    env = RoboTwinEnv(cfg=environment_config(args.robotwin_root.resolve(), official, TASK, len(candidates)),
                      num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None,
                      record_metrics=False)
    try:
        def resolve(requested: int) -> int:
            # No model is imported or called; reset identity is the sole observation.
            _, _, resolved_seed, _ = reset_with_contract(env, requested)
            return int(resolved_seed)
        selected, audit = select_reset_unique_scenes(
            candidates, resolver=resolve, excluded=excluded, count=EXPECTED_GROUPS
        )
    finally:
        env.venv.close(clear_cache=False)
    candidate_contract = {
        "payload": dict(candidate),
        "source": str(candidate_path),
        "source_sha256": sha256(candidate_path),
        "official_runtime_registry": str(official),
        "official_runtime_registry_sha256": sha256(official),
    }
    result = make_seed_manifest(selected=selected, audit=audit, sources=sources,
                                candidate=candidate_contract)
    validate_seed_manifest(result, verify_files=True)
    atomic_json(args.output, result)
    print("V7_SEEDS_PREREGISTERED=" + json.dumps({"output": str(args.output.resolve()),
          "groups": EXPECTED_GROUPS, "labels_read": False,
          "payload_sha256": result["seed_manifest_payload_sha256"]}, sort_keys=True))


if __name__ == "__main__": main()
