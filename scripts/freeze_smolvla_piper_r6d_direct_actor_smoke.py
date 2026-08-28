#!/usr/bin/env python3
"""Freeze a new independent R6d-bound direct-actor simulation preregistration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from execute_smolvla_piper_r6c_simulation_smoke import atomic_json, bind_development_seed_manifest, bind_r6c_preflight
from run_smolvla_piper_r6d_direct_actor_smoke import bind_r6d_simulation_receipt, build_direct_actor_preregistration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6c-manifest", type=Path, required=True)
    parser.add_argument("--r6c-receipt", type=Path, required=True)
    parser.add_argument("--r6d-preregistration", type=Path, required=True)
    parser.add_argument("--r6d-receipt", type=Path, required=True)
    parser.add_argument("--development-seed-manifest", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r6c = bind_r6c_preflight(args.r6c_manifest, args.r6c_receipt)
    r6d = bind_r6d_simulation_receipt(args.r6d_preregistration, args.r6d_receipt)
    seed = bind_development_seed_manifest(args.development_seed_manifest)
    value = build_direct_actor_preregistration(
        r6c=r6c, r6d=r6d, seed=seed, rlinf_root=args.rlinf_root,
        robotwin_root=args.robotwin_root, robotwin_code=args.robotwin_code,
        lerobot_root=args.lerobot_root, model_path=args.model_path,
        vlm_metadata_path=args.vlm_metadata_path, output=args.receipt_output,
    )
    atomic_json(args.output, value)
    print(json.dumps({"status": value["status"], "preregistration_sha256": value["preregistration_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
