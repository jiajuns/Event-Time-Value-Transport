#!/usr/bin/env python3
"""Verify checkpoint schedule, model update, and three-camera deployment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--actor-output", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256(args.source / "model.safetensors")
    if source_sha != args.expected_source_sha:
        raise ValueError("source SHA256 changed")
    final = args.actor_output / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
    if not final.is_dir():
        raise ValueError(f"missing final checkpoint {final}")
    expected_steps = list(range(args.save_freq, args.steps + 1, args.save_freq))
    actual_steps = sorted(int(path.name) for path in (args.actor_output / "checkpoints").iterdir()
                          if path.is_dir() and path.name.isdigit())
    if actual_steps != expected_steps:
        raise ValueError(f"checkpoint schedule mismatch: {actual_steps}")

    source_cfg = json.loads((args.source / "config.json").read_text())
    final_cfg = json.loads((final / "config.json").read_text())
    for key in ("type", "input_features", "output_features", "chunk_size", "n_action_steps"):
        if final_cfg[key] != source_cfg[key]:
            raise ValueError(f"deployment contract changed: {key}")
    final_sha = sha256(final / "model.safetensors")
    if final_sha == source_sha:
        raise ValueError("actor weights did not update")

    processor_files = (
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    processor_hashes = {}
    for filename in processor_files:
        source_hash = sha256(args.source / filename)
        if sha256(final / filename) != source_hash:
            raise ValueError(f"processor changed: {filename}")
        processor_hashes[filename] = source_hash

    preflight = json.loads(args.preflight.read_text())
    if preflight["status"] != "passed" or preflight["method"] != "original_cfc_awr_three_camera":
        raise ValueError("wrong preflight")
    score_receipt = json.loads((args.scores / "receipt.json").read_text())
    result = {
        "status": "passed",
        "method": "original_cfc_awr_three_camera",
        "task_text": args.task,
        "additional_steps": args.steps,
        "save_freq": args.save_freq,
        "source_actor": str(args.source),
        "source_model_sha256": source_sha,
        "actor_output": str(args.actor_output),
        "final_checkpoint": str(final),
        "final_model_sha256": final_sha,
        "processor_sha256": processor_hashes,
        "preflight_sha256": sha256(args.preflight),
        "sidecar_sha256": score_receipt["sidecar_sha256"],
        "training_semantics": "offline original-CfC advantage-weighted regression on logged chunks",
        "three_camera_policy": True,
        "deployment_requires_critic": False,
    }
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
