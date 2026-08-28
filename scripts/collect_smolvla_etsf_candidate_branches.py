#!/usr/bin/env python3
"""Evaluate native SmolVLA candidates on same-seed RoboTwin branches.

Candidate 0 is the frozen SmolVLA baseline: its flow-matching noise is fixed by
the scene seed and policy-query index.  Candidates 1..K-1 differ only in that
noise.  For K>1 every candidate is executed from the same initial simulator
state for the first action chunk; all later chunks use candidate 0.  This makes
the outcomes valid supervision for an action-conditioned ETSF scorer without
fine-tuning the actor or changing the test scenes.

K=1 is a resumable baseline rollout evaluator.  K>1 is a candidate-branch
collector.  No videos or network uploads are performed by this script.

Schema v2's ``candidate_hidden`` is the final 720-D action-expert state after
flow denoising.  It depends on candidate noise and is valid only for the legacy
direct-Q baseline; it is not a shared observation state for an event world
model.  Use ``collect_smolvla_etsf_event_branches.py`` for schema-v5 data.
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
from omegaconf import OmegaConf


DEFAULT_TASK = "move_can_pot"
BODY = "aloha-agilex"
ACTION_DIM = 14
MAX_STEPS = 200
SCHEMA_VERSION = 2


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def scalar_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0])
    return bool(np.asarray(value).reshape(-1)[0])


def load_official_seeds(
    seeds_path: Path, task_name: str, limit: int, offset: int
) -> list[int]:
    data = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in data[task_name]["success_seeds"]]
    selected = seeds[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"requested {limit} seeds at offset {offset}, only found {len(selected)}"
        )
    return selected


def environment_config(
    robotwin_root: Path,
    seeds_path: Path,
    task_name: str,
    episode_num: int,
    max_steps: int,
):
    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": False,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": max_steps,
            "max_episode_steps": max_steps,
            "is_eval": True,
            "assets_path": str(robotwin_root),
            "seeds_path": str(seeds_path),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/etsf_smolvla_video_disabled",
            },
            "task_config": {
                "task_name": task_name,
                "step_lim": max_steps,
                "planner_backend": "mplib",
                "render_freq": 0,
                "episode_num": episode_num,
                "use_seed": False,
                "save_freq": 15,
                "embodiment": [BODY],
                "language_num": 100,
                "domain_randomization": {
                    "random_background": True,
                    "cluttered_table": True,
                    "clean_background_rate": 0.02,
                    "random_head_camera_dis": 0,
                    "random_table_height": 0.03,
                    "random_light": True,
                    "crazy_random_light_rate": 0.02,
                },
                "camera": {
                    "head_camera_type": "D435",
                    "wrist_camera_type": "D435",
                    "collect_head_camera": True,
                    "collect_wrist_camera": True,
                },
                "data_type": {
                    "rgb": True,
                    "third_view": False,
                    "depth": False,
                    "pointcloud": False,
                    "observer": False,
                    "endpose": False,
                    "qpos": True,
                    "mesh_segmentation": False,
                    "actor_segmentation": False,
                },
                "pcd_down_sample_num": 1024,
                "pcd_crop": True,
                "save_path": "/tmp/etsf_smolvla_data_disabled",
                "clear_cache_freq": 8,
                "collect_data": False,
                "eval_video_log": False,
            },
        }
    )


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class ExpertHiddenCapture:
    """Capture the final action-expert representation used by each candidate."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.latest: torch.Tensor | None = None
        self.calls = 0
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is not None:
            self.calls += 1
            self.latest = tensor.detach().float().cpu()

    def reset(self) -> None:
        self.latest = None
        self.calls = 0

    def close(self) -> None:
        self.handle.remove()


def image_chw(image: torch.Tensor) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {tuple(image.shape)}")
    return image.permute(2, 0, 1).contiguous().float().div(255.0)


def raw_policy_input(obs: dict[str, Any], image_keys: list[str]) -> dict[str, Any]:
    main = obs["main_images"][0]
    wrists = obs.get("wrist_images")
    if wrists is None or wrists.shape[1] < 2:
        raise RuntimeError("SmolVLA ALOHA evaluation requires both wrist cameras")
    sources = {
        "observation.images.cam_high": main,
        "observation.images.cam_left_wrist": wrists[0, 0],
        "observation.images.cam_right_wrist": wrists[0, 1],
    }
    fallback = [main, wrists[0, 0], wrists[0, 1]]
    result: dict[str, Any] = {
        "observation.state": obs["states"][0].detach().float().cpu(),
        "task": str(obs["task_descriptions"][0]),
    }
    for index, key in enumerate(image_keys):
        result[key] = image_chw(sources.get(key, fallback[min(index, 2)]))
    if result["observation.state"].numel() != ACTION_DIM:
        raise ValueError(
            f"expected {ACTION_DIM}-D ALOHA state, got {result['observation.state'].numel()}"
        )
    return result


def reset_with_resolved_seed(
    env, requested_seed: int, fixed_instruction: str | None = None
):
    """Reset and expose RoboTwin's silent unstable-seed retry result."""
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
    task = subenv.task
    if not hasattr(task, "ep_num"):
        raise RuntimeError("RoboTwin task does not expose the resolved episode seed")
    observed_instruction = str(obs["task_descriptions"][0])
    if fixed_instruction is not None and observed_instruction != fixed_instruction:
        raise RuntimeError(
            "RoboTwin reset did not preserve the fixed branch instruction"
        )
    return obs, info, int(task.ep_num), observed_instruction


def noise_seed(scene_seed: int, query_index: int, candidate_index: int) -> int:
    # Keep seeds inside the signed 63-bit range accepted by torch.Generator.
    return int(
        (20260826 + int(scene_seed) * 1_000_003 + query_index * 10_007 + candidate_index * 101)
        % (2**63 - 1)
    )


def make_noise(
    config,
    scene_seed: int,
    query_index: int,
    candidate_index: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed(scene_seed, query_index, candidate_index))
    return torch.randn(
        (1, config.chunk_size, config.max_action_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def generate_candidates(
    policy,
    preprocessor,
    postprocessor,
    capture: ExpertHiddenCapture,
    obs: dict[str, Any],
    scene_seed: int,
    query_index: int,
    candidate_count: int,
    device: torch.device,
) -> dict[str, Any]:
    raw = raw_policy_input(obs, list(policy.config.image_features))
    processed = preprocessor(raw)
    chunks: list[torch.Tensor] = []
    hidden: list[torch.Tensor] = []
    hook_calls: list[int] = []
    elapsed: list[float] = []
    for candidate_index in range(candidate_count):
        capture.reset()
        noise = make_noise(
            policy.config, scene_seed, query_index, candidate_index, device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(dict(processed), noise=noise)
        action = postprocessor(normalized)[0].detach().float().cpu()
        elapsed.append(time.perf_counter() - started)
        if action.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"checkpoint emitted {action.shape[-1]} actions, expected {ACTION_DIM}"
            )
        latest = capture.latest
        if latest is None or latest.ndim != 3:
            raise RuntimeError("SmolVLA expert hidden hook did not capture [B,T,D]")
        chunks.append(action)
        hidden.append(latest[0].mean(dim=0))
        hook_calls.append(capture.calls)
    stacked = torch.stack(chunks)
    hidden_stacked = torch.stack(hidden)
    delta = stacked - stacked[0:1]
    return {
        "actions_tensor": stacked,
        "candidate_actions": stacked.numpy().astype(np.float32),
        "candidate_hidden": hidden_stacked.numpy().astype(np.float16),
        "hook_calls": np.asarray(hook_calls, dtype=np.int16),
        "noise_seeds": np.asarray(
            [
                noise_seed(scene_seed, query_index, candidate_index)
                for candidate_index in range(candidate_count)
            ],
            dtype=np.int64,
        ),
        "l2_from_baseline": torch.sqrt(torch.mean(delta.square(), dim=(1, 2)))
        .numpy()
        .astype(np.float32),
        "max_abs_from_baseline": delta.abs()
        .amax(dim=(1, 2))
        .numpy()
        .astype(np.float32),
        "elapsed_seconds": np.asarray(elapsed, dtype=np.float32),
    }


def run_branch(
    policy,
    preprocessor,
    postprocessor,
    capture: ExpertHiddenCapture,
    env,
    scene_seed: int,
    expected_resolved_seed: int,
    fixed_instruction: str,
    first_chunk: torch.Tensor,
    device: torch.device,
    max_steps: int,
    action_exec_steps: int,
) -> dict[str, Any]:
    obs, _, resolved_seed, branch_instruction = reset_with_resolved_seed(
        env, scene_seed, fixed_instruction=fixed_instruction
    )
    if resolved_seed != expected_resolved_seed:
        raise RuntimeError(
            f"non-deterministic seed retry: requested {scene_seed}, "
            f"expected {expected_resolved_seed}, got {resolved_seed}"
        )
    success = False
    done = False
    steps = 0
    query_index = 0
    started = time.time()
    while steps < max_steps and not done:
        if query_index == 0:
            chunk = first_chunk
        else:
            generated = generate_candidates(
                policy,
                preprocessor,
                postprocessor,
                capture,
                obs,
                scene_seed,
                query_index,
                1,
                device,
            )
            chunk = generated["actions_tensor"][0]
        query_index += 1
        for action in chunk[:action_exec_steps]:
            env_action = action.reshape(1, 1, ACTION_DIM)
            obs, _, terminated, truncated, infos = env.step(
                env_action, auto_reset=False
            )
            steps += 1
            success = success or scalar_bool(infos.get("success", [False]))
            done = scalar_bool(terminated) or scalar_bool(truncated)
            if done or steps >= max_steps:
                break
    return {
        "success": success,
        "steps": steps,
        "queries": query_index,
        "wall_seconds": time.time() - started,
        "instruction": branch_instruction,
    }


def save_group(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with h5py.File(temporary, "w") as handle:
        for key in [
            "seed",
            "requested_seed",
            "resolved_seed",
            "task",
            "body",
            "instruction",
            "checkpoint",
        ]:
            handle.attrs[key] = record[key]
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["candidate_generator"] = "native_smolvla_flow_matching"
        handle.attrs["candidate_hidden_contract"] = (
            "action_expert_noise_conditioned_not_shared_observation_state"
        )
        handle.attrs["intervention"] = (
            "candidate_first_executed_prefix_then_fixed_noise_baseline"
        )
        handle.attrs["max_steps"] = record["max_steps"]
        handle.attrs["action_exec_steps"] = record["action_exec_steps"]
        handle.attrs["branch_instruction_consistent"] = bool(
            record["branch_instruction_consistent"]
        )
        for key in [
            "candidate_actions",
            "candidate_hidden",
            "hook_calls",
            "noise_seeds",
            "l2_from_baseline",
            "max_abs_from_baseline",
            "elapsed_seconds",
            "success",
            "steps",
            "queries",
            "wall_seconds",
        ]:
            value = np.asarray(record[key])
            handle.create_dataset(
                key, data=value, compression="gzip" if value.size > 64 else None
            )
        handle.flush()
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--action-exec-steps",
        type=int,
        help="actions executed from each predicted chunk before replanning",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit <= 0 or args.candidate_count <= 0 or args.max_steps <= 0:
        parser.error("--limit, --candidate-count, and --max-steps must be positive")
    if not args.model_path.is_dir() or not args.vlm_metadata_path.is_dir():
        raise FileNotFoundError("model and VLM metadata paths must be local directories")
    if not torch.cuda.is_available():
        raise RuntimeError("SmolVLA RoboTwin evaluation requires the CUDA 4090 host")

    random.seed(20260826)
    np.random.seed(20260826)
    torch.manual_seed(20260826)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    if args.seeds is None:
        seeds = load_official_seeds(seeds_path, args.task, args.limit, args.offset)
    else:
        official = set(load_official_seeds(seeds_path, args.task, 150, 0))
        seeds = [int(seed) for seed in args.seeds]
        invalid = sorted(set(seeds) - official)
        if invalid:
            raise ValueError(f"non-official seeds requested: {invalid}")
        if len(seeds) != len(set(seeds)):
            raise ValueError("--seeds contains duplicates")

    device = torch.device("cuda:0")
    config = PreTrainedConfig.from_pretrained(
        args.model_path, local_files_only=True
    )
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
    if config.action_feature is None or config.action_feature.shape[0] != ACTION_DIM:
        raise ValueError(
            f"checkpoint is not the expected 14-D ALOHA policy: {config.output_features}"
        )
    action_exec_steps = args.action_exec_steps or int(config.n_action_steps)
    if not 1 <= action_exec_steps <= int(config.chunk_size):
        parser.error(
            f"--action-exec-steps must be in [1, {int(config.chunk_size)}]"
        )

    capture = ExpertHiddenCapture(policy.model.vlm_with_expert.lm_expert.norm)
    env = RoboTwinEnv(
        cfg=environment_config(
            args.robotwin_root, seeds_path, args.task, len(seeds), args.max_steps
        ),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output / "groups"
    groups_dir.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "collecting",
        "task": args.task,
        "body": BODY,
        "checkpoint": str(args.model_path),
        "requested_seeds": seeds,
        "candidate_count": args.candidate_count,
        "candidate_generator": "native_smolvla_flow_matching_explicit_fixed_noise",
        "baseline_contract": "candidate_0_fixed_by_scene_seed_and_query_index",
        "intervention": "candidate_first_executed_prefix_then_fixed_noise_baseline",
        "language_contract": "same_instruction_for_initial_query_and_all_candidate_branches",
        "max_steps": args.max_steps,
        "action_exec_steps": action_exec_steps,
        "action_dim": ACTION_DIM,
        "action_chunk": int(config.chunk_size),
        "hidden_dim": int(policy.model.vlm_with_expert.lm_expert.config.hidden_size),
        "candidate_hidden_contract": (
            "action_expert_noise_conditioned_not_shared_observation_state"
        ),
        "groups": [],
    }
    resolved_seeds_seen: set[int] = set()
    try:
        for group_index, seed in enumerate(seeds):
            path = groups_dir / f"group_{group_index:03d}_seed_{seed}.hdf5"
            if path.exists() and not args.overwrite:
                with h5py.File(path, "r") as handle:
                    successes = handle["success"][:].astype(bool).tolist()
                    steps = handle["steps"][:].astype(int).tolist()
                    resolved_seed = int(handle.attrs.get("resolved_seed", seed))
                if resolved_seed in resolved_seeds_seen:
                    raise RuntimeError(
                        f"duplicate resolved scene seed {resolved_seed}; "
                        "choose a replacement official seed"
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
                        "steps": steps,
                        "status": "existing",
                    }
                )
                continue

            obs, _, resolved_seed, instruction = reset_with_resolved_seed(env, seed)
            if resolved_seed in resolved_seeds_seen:
                raise RuntimeError(
                    f"requested seed {seed} resolves to duplicate scene "
                    f"{resolved_seed}; choose a replacement official seed"
                )
            resolved_seeds_seen.add(resolved_seed)
            generated = generate_candidates(
                policy,
                preprocessor,
                postprocessor,
                capture,
                obs,
                seed,
                0,
                args.candidate_count,
                device,
            )
            chunks = generated.pop("actions_tensor")
            outcomes = [
                run_branch(
                    policy,
                    preprocessor,
                    postprocessor,
                    capture,
                    env,
                    seed,
                    resolved_seed,
                    instruction,
                    chunk,
                    device,
                    args.max_steps,
                    action_exec_steps,
                )
                for chunk in chunks
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
                    outcome["instruction"] == instruction for outcome in outcomes
                ),
                "checkpoint": str(args.model_path),
                "max_steps": args.max_steps,
                "action_exec_steps": action_exec_steps,
                "success": np.asarray(
                    [outcome["success"] for outcome in outcomes], dtype=bool
                ),
                "steps": np.asarray(
                    [outcome["steps"] for outcome in outcomes], dtype=np.int32
                ),
                "queries": np.asarray(
                    [outcome["queries"] for outcome in outcomes], dtype=np.int16
                ),
                "wall_seconds": np.asarray(
                    [outcome["wall_seconds"] for outcome in outcomes],
                    dtype=np.float32,
                ),
            }
            if not record["branch_instruction_consistent"]:
                raise RuntimeError(
                    f"candidate branches changed instruction for seed {seed}"
                )
            save_group(path, record)
            item = {
                "index": group_index,
                "seed": seed,
                "requested_seed": seed,
                "resolved_seed": resolved_seed,
                "path": path.name,
                "success": record["success"].tolist(),
                "steps": record["steps"].tolist(),
                "l2_from_baseline": record["l2_from_baseline"].tolist(),
                "wall_seconds": float(record["wall_seconds"].sum()),
                "status": "collected",
            }
            manifest["groups"].append(item)
            manifest["completed"] = len(manifest["groups"])
            atomic_json(args.output / "manifest.json", manifest)
            print("COLLECTED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        capture.close()
        env.venv.close(clear_cache=False)

    successes = np.asarray(
        [item["success"] for item in manifest["groups"]], dtype=np.int64
    )
    manifest["status"] = "complete"
    manifest["completed"] = len(manifest["groups"])
    manifest["candidate_successes"] = successes.sum(axis=0).tolist()
    manifest["candidate_success_rates"] = successes.mean(axis=0).tolist()
    manifest["oracle_successes"] = int(successes.max(axis=1).sum())
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
                key: manifest[key]
                for key in [
                    "completed",
                    "candidate_successes",
                    "oracle_successes",
                    "groups_with_outcome_variation",
                ]
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
