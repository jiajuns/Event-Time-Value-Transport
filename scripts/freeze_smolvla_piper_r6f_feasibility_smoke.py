#!/usr/bin/env python3
"""Freeze independent R6f fixed-candidate feasibility simulation authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_smolvla_piper_r6d_direct_actor_smoke import _load_and_recompute_preregistration, atomic_json
from run_smolvla_piper_r6f_feasibility_smoke import bind_r6e_preregistration, build_feasibility_preregistration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6e-preregistration", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r6e = bind_r6e_preregistration(args.r6e_preregistration)
    r6e_prereg, _, _, _ = _load_and_recompute_preregistration(args.r6e_preregistration)
    value = build_feasibility_preregistration(
        r6e=r6e,
        r6e_preregistration=r6e_prereg,
        output=args.receipt_output,
    )
    atomic_json(args.output, value)
    print(json.dumps({
        "status": value["status"],
        "preregistration_sha256": value["preregistration_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
