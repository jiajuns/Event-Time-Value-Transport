#!/usr/bin/env python3
"""Verify SmolVLA candidates and the noise-independent ETSF VLM state hook.

This is an interface smoke test, not a task-success evaluation.  It deliberately
loads the base policy without its original VLM weights, because the complete
SmolVLA policy checkpoint immediately replaces the initialized parameters.  A
small local SmolVLM metadata directory is still required to construct the model
and tokenizer without network access.

The hook targets the contextualized VLM prefix state before flow denoising.  It
does not expose or accept the candidate-specific 720-D action-expert hidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def reject_fresh_path(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if any("fresh" in part.casefold() for part in resolved.parts):
        raise ValueError(f"{role} must not reference Fresh data")
    return resolved


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path = reject_fresh_path(path, "array output")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def make_image(path: Path | None, height: int, width: int) -> torch.Tensor:
    if path is not None:
        image = Image.open(path).convert("RGB").resize((width, height))
        array = np.asarray(image, dtype=np.float32) / 255.0
    else:
        yy, xx = np.mgrid[0:height, 0:width]
        array = np.stack(
            [
                xx / max(width - 1, 1),
                yy / max(height - 1, 1),
                (xx + yy) / max(width + height - 2, 1),
            ],
            axis=-1,
        ).astype(np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def pairwise_l2(chunks: torch.Tensor) -> np.ndarray:
    flat = chunks.float().flatten(1)
    return torch.cdist(flat, flat, p=2).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move the can into the pot")
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        help="repeat for each configured camera; one image is replicated across cameras",
    )
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument(
        "--candidate-actions-output",
        type=Path,
        help="optional safe .npy output containing the four unnormalized Aloha chunks",
    )
    parser.add_argument(
        "--shared-prefixes-output",
        type=Path,
        help="optional safe .npy output containing the four shared VLM prefix states",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    if args.candidate_count < 2:
        raise ValueError("candidate-count must be at least 2")
    if (args.candidate_actions_output is None) != (args.shared_prefixes_output is None):
        raise ValueError("both array outputs must be supplied together")
    if not args.model_path.is_dir() or not args.vlm_metadata_path.is_dir():
        raise FileNotFoundError("model-path and vlm-metadata-path must be local directories")
    if not torch.cuda.is_available():
        raise RuntimeError("this smoke test is intended for the CUDA 4090 environment")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import OBS_STATE
    from collect_smolvla_etsf_event_branches import (
        SHARED_STATE_ANCHOR,
        SHARED_STATE_SOURCE,
        resolve_shared_prefix_capture,
    )

    device = torch.device("cuda:0")
    # Parse through the choice registry. Calling the concrete subclass directly
    # makes draccus treat the serialized ``type`` discriminator as an unknown
    # dataclass field in LeRobot 0.4.4.
    config = PreTrainedConfig.from_pretrained(args.model_path, local_files_only=True)
    config.device = str(device)
    config.vlm_model_name = str(args.vlm_metadata_path)
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        args.model_path,
        config=config,
        local_files_only=True,
        strict=True,
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.model_path),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_metadata_path)},
        },
    )
    if config.robot_state_feature is None:
        raise ValueError("SmolVLA checkpoint does not define an observation.state feature")
    state_dim = args.state_dim or config.robot_state_feature.shape[0]
    if state_dim > config.max_state_dim:
        raise ValueError(f"state-dim {state_dim} exceeds max_state_dim {config.max_state_dim}")
    raw: dict[str, Any] = {
        OBS_STATE: torch.zeros(state_dim, dtype=torch.float32),
        "task": args.task,
    }
    image_paths: list[Path | None] = args.image or [None]
    for camera_index, key in enumerate(config.image_features):
        source = image_paths[min(camera_index, len(image_paths) - 1)]
        raw[key] = make_image(source, args.height, args.width)
    batch = preprocessor(raw)

    # The VLM final norm is called for the prefix KV-cache fill and not for the
    # action-expert denoise steps.  The resolver rejects padded/unknown layouts.
    capture = resolve_shared_prefix_capture(policy)
    torch.cuda.reset_peak_memory_stats(device)
    chunks = []
    shared_states = []
    hook_calls = []
    elapsed = []
    try:
        for candidate_index in range(args.candidate_count):
            policy.reset()
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + candidate_index)
            noise = torch.randn(
                (1, config.chunk_size, config.max_action_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            capture.reset()
            started = time.perf_counter()
            normalized = policy.predict_action_chunk(dict(batch), noise=noise)
            chunk = postprocessor(normalized)
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
            chunks.append(chunk[0].detach().float().cpu())
            shared_states.append(capture.consume())
            hook_calls.append(capture.calls)
    finally:
        capture.close()

    stacked = torch.stack(chunks)
    shared_stack = torch.stack(shared_states).float()
    shared_reference = shared_stack[0:1].expand_as(shared_stack)
    shared_max_abs_delta = (shared_stack - shared_reference).abs().amax(dim=1)
    if not torch.equal(shared_stack, shared_reference):
        raise RuntimeError(
            "flow noise changed the shared VLM prefix state: "
            f"max_abs_delta={float(shared_max_abs_delta.max())}"
        )
    distances = pairwise_l2(stacked)
    upper = distances[np.triu_indices(args.candidate_count, 1)]
    identical_pairs = int(np.isclose(upper, 0.0, atol=1e-7).sum())
    action_array = stacked.numpy()
    prefix_array = shared_stack.numpy()
    array_outputs = None
    if args.candidate_actions_output is not None:
        action_path = reject_fresh_path(args.candidate_actions_output, "candidate actions output")
        prefix_path = reject_fresh_path(args.shared_prefixes_output, "shared prefixes output")
        atomic_npy(action_path, action_array)
        atomic_npy(prefix_path, prefix_array)
        array_outputs = {
            "candidate_actions": str(action_path),
            "candidate_actions_array_sha256": array_sha256(action_array),
            "shared_prefixes": str(prefix_path),
            "shared_prefix_array_sha256": array_sha256(prefix_array[0]),
        }
    summary = {
        "schema_version": 1,
        "experiment_type": "interface_smoke_not_task_success",
        "model_path": str(args.model_path),
        "vlm_metadata_path": str(args.vlm_metadata_path),
        "task": args.task,
        "image_source": (
            [str(path) for path in image_paths]
            if args.image is not None
            else "deterministic_gradient"
        ),
        "candidate_generator": "native_smolvla_flow_matching_explicit_noise",
        "preprocessing": "checkpoint_preprocessor_and_postprocessor",
        "model_config_observation_state_dim": int(config.robot_state_feature.shape[0]),
        "runtime_observation_state_dim": int(state_dim),
        "runtime_preprocessed_observation_state_shape": list(batch[OBS_STATE].shape),
        "state_dimension_override_used": args.state_dim is not None,
        "candidate_count": args.candidate_count,
        "candidate_shape": list(stacked.shape),
        "seed_base": args.seed,
        "pairwise_l2": distances.tolist(),
        "pairwise_l2_min_non_diagonal": float(upper.min()),
        "pairwise_l2_mean_non_diagonal": float(upper.mean()),
        "identical_candidate_pairs": identical_pairs,
        "per_element_candidate_std_mean": float(stacked.float().std(dim=0).mean()),
        "action_abs_max": float(stacked.float().abs().max()),
        "elapsed_seconds_per_candidate": elapsed,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "etsf_shared_state_hook": {
            "module": SHARED_STATE_SOURCE,
            "anchor": SHARED_STATE_ANCHOR,
            "shape": list(shared_stack.shape),
            "feature_dim": int(shared_stack.shape[-1]),
            "hook_calls_per_candidate": hook_calls,
            "max_abs_delta_from_candidate_0": shared_max_abs_delta.tolist(),
            "bit_exact_across_noise_candidates": True,
            "candidate_specific_expert_hidden_saved": False,
            "status": "verified",
        },
        "native_multi_candidate_verified": identical_pairs == 0 and float(upper.min()) > 0.0,
        "array_outputs": array_outputs,
        "task_success_claimed": False,
    }
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
