#!/usr/bin/env python3
"""Freeze and validate a real-artifact SmolVLA(Aloha) -> Piper preflight.

This command is deliberately offline and data-blind.  It hashes the exact actor
files, the real RoboTwin body ``config.yml``/URDF pairs, and an already-produced
CUDA forward probe.  It does not import a policy, start RoboTwin, call
``env.step``, read outcomes, or authorize a transfer-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from verify_smolvla_piper_zero_shot_preflight import (
    ACTOR_ID,
    FORMAT,
    expected_slot_mapping_contract,
    expected_state_dimension_resolution,
    file_sha256,
    reject_fresh_path,
    run_preflight,
)


def artifact(path: Path, role: str) -> dict[str, str]:
    resolved = reject_fresh_path(path, role)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def build_manifest(
    *,
    model_path: Path,
    robotwin_root: Path,
    forward_probe_receipt: Path,
    candidate_actions: Path,
    shared_prefixes: Path,
    source_image: Path,
) -> dict[str, Any]:
    model = reject_fresh_path(model_path, "model_path")
    robotwin = reject_fresh_path(robotwin_root, "robotwin_root")
    if not model.is_dir():
        raise NotADirectoryError(model)
    if not robotwin.is_dir():
        raise NotADirectoryError(robotwin)

    piper = robotwin / "assets" / "embodiments" / "piper"
    aloha = robotwin / "assets" / "embodiments" / "aloha-agilex"
    static_paths = {
        "checkpoint_config": model / "config.json",
        "model_weights": model / "model.safetensors",
        "train_config": model / "train_config.json",
        "policy_preprocessor": model / "policy_preprocessor.json",
        "policy_postprocessor": model / "policy_postprocessor.json",
        "preprocessor_stats": (
            model / "policy_preprocessor_step_5_normalizer_processor.safetensors"
        ),
        "postprocessor_stats": (
            model / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        ),
        "piper_body_config": piper / "config.yml",
        "aloha_body_config": aloha / "config.yml",
        "piper_urdf": piper / "piper.urdf",
        "aloha_urdf": aloha / "urdf" / "arx5_description_isaac.urdf",
    }
    probe_paths = {
        "forward_probe_receipt": forward_probe_receipt,
        "candidate_actions": candidate_actions,
        "shared_prefixes": shared_prefixes,
        "source_image": source_image,
    }
    return {
        "format": FORMAT,
        "actor_id": ACTOR_ID,
        "source_body": "aloha",
        "target_body": "piper",
        "artifacts": {
            name: artifact(path, f"artifacts.{name}")
            for name, path in static_paths.items()
        },
        "probe_artifacts": {
            name: artifact(path, f"probe_artifacts.{name}")
            for name, path in probe_paths.items()
        },
        "slot_mapping_contract": expected_slot_mapping_contract(),
        "state_dimension_resolution": expected_state_dimension_resolution(),
        "capability_contract": {
            "fresh_inputs_allowed": False,
            "environment_step_allowed": False,
            "outcome_inputs_allowed": False,
            "execution_authorized": False,
            "transfer_claim_authorized": False,
            "maximum_authorization": "forward_only",
        },
    }


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_pair(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> None:
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for destination, payload in (
            (manifest_path, manifest),
            (receipt_path, receipt),
        ):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
            )
            temporary = Path(name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_serialized_json(payload))
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_paths[0], manifest_path)
        os.replace(temporary_paths[1], receipt_path)
        manifest_path.chmod(0o444)
        receipt_path.chmod(0o444)
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--forward-probe-receipt", type=Path, required=True)
    parser.add_argument("--candidate-actions", type=Path, required=True)
    parser.add_argument("--shared-prefixes", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args()

    output_manifest = reject_fresh_path(args.output_manifest, "output_manifest")
    output_receipt = reject_fresh_path(args.output_receipt, "output_receipt")
    manifest = build_manifest(
        model_path=args.model_path,
        robotwin_root=args.robotwin_root,
        forward_probe_receipt=args.forward_probe_receipt,
        candidate_actions=args.candidate_actions,
        shared_prefixes=args.shared_prefixes,
        source_image=args.source_image,
    )
    receipt = run_preflight(manifest)
    manifest_bytes = _serialized_json(manifest)
    receipt = {
        **receipt,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "freezer_implementation_sha256": file_sha256(Path(__file__)),
    }
    _write_pair(output_manifest, manifest, output_receipt, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "authorization": receipt["authorization"],
                "manifest_file_sha256": receipt["manifest_file_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
