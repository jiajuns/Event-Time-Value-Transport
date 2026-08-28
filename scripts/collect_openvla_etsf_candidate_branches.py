#!/usr/bin/env python3
"""Collect same-seed OpenVLA-OFT candidate branches for action-conditioned ETSF.

For every RoboTwin reset seed this collector creates one deterministic OpenVLA
action chunk and several chunks sampled from the model's own discrete action
token distribution.  Each candidate is then evaluated from the *same* initial
simulator state: the candidate controls the first chunk and the frozen,
deterministic OpenVLA policy controls the rest of the episode.  The resulting
outcomes provide real simulator supervision for Q_ETSF(h, action_chunk).

This script never changes or fine-tunes the OpenVLA actor.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from collect_openvla_etsf_rollouts import (
    ACTION_DIM,
    BODY,
    CHUNK,
    DEFAULT_TASK,
    MAX_STEPS,
    atomic_json,
    environment_config,
    install_hidden_hook,
    load_official_seeds,
    model_config,
    predict,
    scalar_bool,
)


SCHEMA_VERSION = 2
GRIPPER_DIMS = (6, 13)


def sampled_action(model, obs, temperature: float, top_k: int):
    actions, info = model.predict_action_batch(
        env_obs=obs,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        calulate_values=False,
    )
    logprobs = info["prev_logprobs"].detach().float().cpu().numpy()
    return actions, logprobs


def generate_candidates(
    model,
    obs,
    capture: dict[str, torch.Tensor],
    seed: int,
    blends: list[float],
    temperature: float,
    top_k: int,
    preserve_grippers: bool,
    action_scale: torch.Tensor,
) -> dict[str, Any]:
    with torch.inference_mode():
        baseline, baseline_info = model.predict_action_batch(
            env_obs=obs,
            do_sample=False,
            temperature=1.0,
            top_k=-1,
            calulate_values=False,
        )
    initial_hidden = capture["last_hidden_states"]
    initial_anchor = initial_hidden[
        :, -model.action_dim * model.num_action_chunks - 1
    ][0].float().cpu().numpy().astype(np.float16)
    baseline_logprobs = (
        baseline_info["prev_logprobs"].detach().float().cpu().numpy()
    )

    candidates = [baseline[0].detach().clone()]
    source_samples = [baseline[0].detach().clone()]
    source_logprobs = [baseline_logprobs[0]]
    names = ["deterministic"]
    for index, blend in enumerate(blends, start=1):
        # Sampling is deterministic for a (scene seed, candidate index) pair,
        # without touching the simulator's NumPy/Python random streams.
        torch.manual_seed(20260826 + int(seed) * 17 + index)
        with torch.inference_mode():
            sampled, logprobs = sampled_action(model, obs, temperature, top_k)
        source = sampled[0].detach()
        candidate = baseline[0] + blend * (source - baseline[0])
        if preserve_grippers:
            candidate[:, list(GRIPPER_DIMS)] = baseline[0, :, list(GRIPPER_DIMS)]
        candidates.append(candidate)
        source_samples.append(source.clone())
        source_logprobs.append(logprobs[0])
        names.append(f"sample_blend_{blend:.3f}")

    candidate_tensor = torch.stack(candidates)
    source_tensor = torch.stack(source_samples)
    delta = candidate_tensor - candidate_tensor[0:1]
    normalized_delta = delta / action_scale.to(delta.device)[None, None, :]
    return {
        "initial_hidden": initial_anchor,
        "candidate_names": names,
        "candidate_actions_tensor": candidate_tensor,
        "candidate_actions": candidate_tensor.float().cpu().numpy().astype(np.float32),
        "source_sampled_actions": source_tensor.float().cpu().numpy().astype(np.float32),
        "source_logprobs": np.stack(source_logprobs).astype(np.float32),
        "l2_from_baseline": torch.sqrt(torch.mean(delta.square(), dim=(1, 2)))
        .cpu()
        .numpy()
        .astype(np.float32),
        "normalized_l2_from_baseline": torch.sqrt(
            torch.mean(normalized_delta.square(), dim=(1, 2))
        )
        .cpu()
        .numpy()
        .astype(np.float32),
        "max_abs_from_baseline": delta.abs()
        .amax(dim=(1, 2))
        .cpu()
        .numpy()
        .astype(np.float32),
    }


def reference_action_scale(root: Path | None, device: torch.device) -> torch.Tensor:
    """Compute stable per-dimension scales from frozen baseline rollouts."""
    if root is None:
        return torch.ones(ACTION_DIM, device=device)
    files = sorted((root / "episodes").glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no reference rollout episodes found under {root}")
    count = 0
    total = np.zeros(ACTION_DIM, dtype=np.float64)
    squared = np.zeros(ACTION_DIM, dtype=np.float64)
    for path in files:
        with h5py.File(path, "r") as handle:
            actions = handle["action_chunks"][:].astype(np.float64).reshape(-1, ACTION_DIM)
        count += len(actions)
        total += actions.sum(axis=0)
        squared += np.square(actions).sum(axis=0)
    variance = np.maximum(squared / count - np.square(total / count), 1e-4)
    return torch.as_tensor(np.sqrt(variance), device=device, dtype=torch.float32)


def reset_with_contract(env, requested_seed: int, fixed_instruction: str | None = None):
    if fixed_instruction is None:
        obs, info = env.reset(env_seeds=[requested_seed])
        subenv = env.venv.envs[0]
    else:
        if not env.venv.envs:
            env.venv._init_envs()
        subenv = env.venv.envs[0]
        original_create_instruction = subenv.create_instruction
        subenv.create_instruction = lambda: fixed_instruction
        try:
            obs, info = env.reset(env_seeds=[requested_seed])
        finally:
            subenv.create_instruction = original_create_instruction
    resolved_seed = int(subenv.task.ep_num)
    observed_instruction = str(obs["task_descriptions"][0])
    if fixed_instruction is not None and observed_instruction != fixed_instruction:
        raise RuntimeError("RoboTwin did not preserve the fixed branch instruction")
    return obs, info, resolved_seed, observed_instruction


def evaluate_first_chunk_branch(
    model,
    env,
    seed: int,
    expected_resolved_seed: int,
    fixed_instruction: str,
    first_chunk: torch.Tensor,
):
    obs, _, resolved_seed, branch_instruction = reset_with_contract(
        env, seed, fixed_instruction=fixed_instruction
    )
    if resolved_seed != expected_resolved_seed:
        raise RuntimeError(
            f"non-deterministic seed retry: requested {seed}, "
            f"expected {expected_resolved_seed}, got {resolved_seed}"
        )
    success = False
    done = False
    steps = 0
    query_count = 0
    started = time.time()

    while steps < MAX_STEPS and not done:
        if query_count == 0:
            chunk = first_chunk.unsqueeze(0)
        else:
            with torch.inference_mode():
                chunk = predict(model, obs)
        query_count += 1
        for action_index in range(chunk.shape[1]):
            action = chunk[:, action_index : action_index + 1]
            obs, _, terminated, truncated, infos = env.step(action, auto_reset=False)
            steps += 1
            success = success or scalar_bool(infos.get("success", [False]))
            done = scalar_bool(terminated) or scalar_bool(truncated)
            if done or steps >= MAX_STEPS:
                break
    return {
        "success": success,
        "steps": steps,
        "queries": query_count,
        "wall_seconds": time.time() - started,
        "instruction": branch_instruction,
    }


def save_group(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as handle:
        for key in [
            "seed",
            "requested_seed",
            "resolved_seed",
            "task",
            "body",
            "instruction",
            "temperature",
            "top_k",
            "preserve_grippers",
        ]:
            handle.attrs[key] = record[key]
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["intervention"] = "candidate_first_chunk_then_deterministic_actor"
        handle.attrs["branch_instruction_consistent"] = bool(
            record["branch_instruction_consistent"]
        )
        handle.create_dataset("initial_hidden", data=record["initial_hidden"], compression="gzip")
        handle.create_dataset("candidate_names", data=np.asarray(record["candidate_names"], dtype=object), dtype=strings)
        for key in [
            "candidate_actions",
            "source_sampled_actions",
            "source_logprobs",
            "l2_from_baseline",
            "normalized_l2_from_baseline",
            "max_abs_from_baseline",
            "success",
            "steps",
            "queries",
            "wall_seconds",
        ]:
            value = np.asarray(record[key])
            compression = "gzip" if value.size > 64 else None
            handle.create_dataset(key, data=value, compression=compression)
        handle.flush()
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--seeds-file",
        type=Path,
        help="JSON list, or a shadow split manifest selected with --seeds-key",
    )
    parser.add_argument("--seeds-key", choices=["train", "validation", "test"])
    parser.add_argument("--reference-rollouts", type=Path)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--allow-sampled-grippers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.seeds is not None and args.seeds_file is not None:
        parser.error("--seeds and --seeds-file are mutually exclusive")
    if args.seeds_key is not None and args.seeds_file is None:
        parser.error("--seeds-key requires --seeds-file")

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k == 0 or args.top_k < -1:
        parser.error("--top-k must be -1 or positive")
    if not args.blends or any(not 0 < value <= 1 for value in args.blends):
        parser.error("--blends values must be in (0, 1]")

    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))

    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openvla_oft.official import get_model

    args.output.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output / "groups"
    groups_dir.mkdir(exist_ok=True)
    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    if args.seeds_file is not None:
        source = json.loads(args.seeds_file.read_text(encoding="utf-8"))
        if args.seeds_key is not None:
            source = source[args.seeds_key]
        if not isinstance(source, list):
            raise ValueError("seed file selection must be a JSON list")
        seeds = [
            int(item["seed"] if isinstance(item, dict) else item) for item in source
        ]
    elif args.seeds is None:
        seeds = load_official_seeds(seeds_path, args.task, args.limit, args.offset)
    else:
        seeds = [int(seed) for seed in args.seeds]
    official = set(load_official_seeds(seeds_path, args.task, 150, 0))
    invalid = sorted(set(seeds) - official)
    if invalid:
        raise ValueError(f"non-official seeds requested: {invalid}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed selection is empty or contains duplicates")
    unnorm_key = args.unnorm_key or f"{args.task}_1k"

    device = torch.device("cuda:0")
    model = (
        get_model(model_config(args.model_path, unnorm_key), torch_dtype=torch.bfloat16)
        .eval()
        .to(device)
    )
    capture = install_hidden_hook(model)
    action_scale = reference_action_scale(args.reference_rollouts, device)
    env = RoboTwinEnv(
        cfg=environment_config(args.robotwin_root, seeds_path, args.task, len(seeds)),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "collecting",
        "collector_seed": 20260826,
        "task": args.task,
        "body": BODY,
        "model_path": str(args.model_path),
        "unnorm_key": unnorm_key,
        "requested_seeds": seeds,
        "candidate_count": 1 + len(args.blends),
        "blends": args.blends,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "preserve_grippers": not args.allow_sampled_grippers,
        "intervention": "candidate_first_chunk_then_deterministic_actor",
        "language_contract": "same_instruction_for_initial_query_and_all_candidate_branches",
        "reference_rollouts": str(args.reference_rollouts) if args.reference_rollouts else None,
        "action_scale": action_scale.cpu().numpy().tolist(),
        "groups": [],
    }
    resolved_seeds_seen: set[int] = set()
    try:
        for group_index, seed in enumerate(seeds):
            path = groups_dir / f"group_{group_index:03d}_seed_{seed}.hdf5"
            if path.exists() and not args.overwrite:
                with h5py.File(path, "r") as handle:
                    successes = handle["success"][:].astype(bool).tolist()
                    resolved_seed = int(handle.attrs.get("resolved_seed", seed))
                if resolved_seed in resolved_seeds_seen:
                    raise RuntimeError(
                        f"duplicate resolved scene {resolved_seed}; choose a replacement seed"
                    )
                resolved_seeds_seen.add(resolved_seed)
                manifest["groups"].append(
                    {
                        "index": group_index,
                        "seed": seed,
                        "requested_seed": seed,
                        "resolved_seed": resolved_seed,
                        "path": path.name,
                        "success": successes,
                        "status": "existing",
                    }
                )
                continue

            obs, _, resolved_seed, instruction = reset_with_contract(env, seed)
            if resolved_seed in resolved_seeds_seen:
                raise RuntimeError(
                    f"requested seed {seed} resolves to duplicate scene {resolved_seed}"
                )
            resolved_seeds_seen.add(resolved_seed)
            generated = generate_candidates(
                model,
                obs,
                capture,
                seed,
                args.blends,
                args.temperature,
                args.top_k,
                not args.allow_sampled_grippers,
                action_scale,
            )
            outcomes = [
                evaluate_first_chunk_branch(
                    model, env, seed, resolved_seed, instruction, candidate
                )
                for candidate in generated.pop("candidate_actions_tensor")
            ]
            record = {
                **generated,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "task": args.task,
                "body": BODY,
                "instruction": instruction,
                "branch_instruction_consistent": all(
                    item["instruction"] == instruction for item in outcomes
                ),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "preserve_grippers": not args.allow_sampled_grippers,
                "success": np.asarray([item["success"] for item in outcomes], dtype=bool),
                "steps": np.asarray([item["steps"] for item in outcomes], dtype=np.int32),
                "queries": np.asarray([item["queries"] for item in outcomes], dtype=np.int32),
                "wall_seconds": np.asarray([item["wall_seconds"] for item in outcomes], dtype=np.float32),
            }
            if not record["branch_instruction_consistent"]:
                raise RuntimeError(f"candidate branches changed instruction for seed {seed}")
            save_group(path, record)
            item = {
                "index": group_index,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "path": path.name,
                "candidate_names": record["candidate_names"],
                "success": record["success"].tolist(),
                "steps": record["steps"].tolist(),
                "l2_from_baseline": record["l2_from_baseline"].tolist(),
                "normalized_l2_from_baseline": record["normalized_l2_from_baseline"].tolist(),
                "wall_seconds": float(record["wall_seconds"].sum()),
                "status": "collected",
            }
            manifest["groups"].append(item)
            manifest["completed"] = len(manifest["groups"])
            atomic_json(args.output / "manifest.json", manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        env.venv.close(clear_cache=False)

    manifest["status"] = "complete"
    manifest["completed"] = len(manifest["groups"])
    successes = np.asarray(
        [item["success"] for item in manifest["groups"]], dtype=np.int64
    )
    manifest["candidate_successes"] = successes.sum(axis=0).tolist()
    manifest["candidate_success_rates"] = successes.mean(axis=0).tolist()
    manifest["groups_with_outcome_variation"] = int(
        sum(len(set(item["success"])) > 1 for item in manifest["groups"])
    )
    manifest["resolved_seeds"] = [
        int(item["resolved_seed"]) for item in manifest["groups"]
    ]
    atomic_json(args.output / "manifest.json", manifest)
    print(
        "COLLECTION_COMPLETE="
        + json.dumps(
            {
                "completed": manifest["completed"],
                "candidate_successes": manifest["candidate_successes"],
                "groups_with_outcome_variation": manifest["groups_with_outcome_variation"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
