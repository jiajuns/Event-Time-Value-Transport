#!/usr/bin/env python3
"""Freeze one independent simulation-only R6c -> Piper short-smoke plan.

The freezer performs only artifact authentication and contract construction. It
does not import RoboTwin, initialize CUDA, reset an environment, execute an
action, inspect Fresh data, or observe an outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execute_smolvla_piper_r6c_simulation_smoke import (
    MAX_SMOKE_STEPS,
    atomic_json,
    bind_development_seed_manifest,
    bind_r6c_preflight,
    build_simulation_preregistration,
    load_r6c_mapped_candidate,
)
from verify_smolvla_piper_zero_shot_preflight import reject_fresh_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--development-seed-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--step-limit", type=int, default=4)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()

    if not args.acknowledge_simulation_only:
        parser.error("--acknowledge-simulation-only is mandatory")
    if not 1 <= args.step_limit <= MAX_SMOKE_STEPS:
        parser.error(f"--step-limit must lie in [1,{MAX_SMOKE_STEPS}]")
    output = reject_fresh_path(args.output, "preregistration output")
    if output.exists():
        raise FileExistsError(output)
    if reject_fresh_path(args.run_output, "run output").exists():
        raise FileExistsError(args.run_output)

    binding = bind_r6c_preflight(args.preflight_manifest, args.preflight_receipt)
    seed_contract = bind_development_seed_manifest(args.development_seed_manifest)
    _, candidate_contract = load_r6c_mapped_candidate(binding, args.candidate_index)
    preregistration = build_simulation_preregistration(
        binding=binding,
        seed_contract=seed_contract,
        candidate_contract=candidate_contract,
        rlinf_root=args.rlinf_root,
        robotwin_root=args.robotwin_root,
        robotwin_code=args.robotwin_code,
        output=args.run_output,
        candidate_index=args.candidate_index,
        step_limit=args.step_limit,
    )
    atomic_json(output, preregistration)
    print(
        json.dumps(
            {
                "status": preregistration["status"],
                "preregistration_sha256": preregistration[
                    "preregistration_sha256"
                ],
                "simulation_execution_authorized": True,
                "real_robot_execution_authorized": False,
                "transfer_claim_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
