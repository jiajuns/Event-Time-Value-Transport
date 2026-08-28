#!/usr/bin/env python3
"""Run same-seed OpenVLA vs guarded action-conditioned ETSF evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from collect_openvla_etsf_candidate_branches import (
    evaluate_first_chunk_branch,
    generate_candidates,
    reference_action_scale,
)
from collect_openvla_etsf_rollouts import (
    BODY,
    atomic_json,
    environment_config,
    install_hidden_hook,
    load_official_seeds,
    model_config,
)
from train_openvla_etsf_action_q import ActionConditionedETSF


def load_baseline_results(root: Path) -> dict[int, dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("baseline rollout manifest is not complete")
    result = {}
    for item in manifest["episodes"]:
        path = root / "episodes" / item["path"]
        with h5py.File(path, "r") as handle:
            seed = int(handle.attrs["seed"])
            result[seed] = {
                "success": bool(handle.attrs["success"]),
                "steps": int(handle.attrs["steps"]),
                "path": str(path),
            }
    return result


def load_action_q(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    semantic_state = {
        key.removeprefix("semantic."): value
        for key, value in state.items()
        if key.startswith("semantic.")
    }
    model = ActionConditionedETSF(
        semantic_state,
        state["action_mean"],
        state["action_std"],
        state["feature_mean"],
        state["feature_std"],
    )
    model.load_state_dict(state)
    return model.eval().to(device), checkpoint


@torch.no_grad()
def score_candidates(
    critic: ActionConditionedETSF,
    initial_hidden: np.ndarray,
    actions: torch.Tensor,
    source_logprobs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(critic.parameters()).device
    actions = actions.to(device)
    count = len(actions)
    hidden = torch.as_tensor(initial_hidden, device=device, dtype=torch.float32)[None].expand(count, -1)
    baseline = actions[0:1].expand(count, -1, -1)
    logprobs = torch.as_tensor(source_logprobs, device=actions.device, dtype=torch.float32)
    logits = critic(hidden, actions.float(), baseline.float(), logprobs)
    return logits.cpu().numpy(), torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--baseline-rollouts", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="move_can_pot")
    parser.add_argument("--unnorm-key")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--blends", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--minimum-probability-margin", type=float)
    parser.add_argument("--maximum-normalized-distance", type=float)
    parser.add_argument("--allow-developmental", action="store_true")
    args = parser.parse_args()

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

    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    official = set(load_official_seeds(seeds_path, args.task, 150, 0))
    invalid = sorted(set(args.seeds) - official)
    if invalid or len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f"invalid or duplicate evaluation seeds: {invalid}")
    baseline_results = load_baseline_results(args.baseline_rollouts)
    missing = sorted(set(args.seeds) - set(baseline_results))
    if missing:
        raise KeyError(f"evaluation seeds missing from frozen baseline rollouts: {missing}")

    device = torch.device("cuda:0")
    critic, q_checkpoint = load_action_q(args.q_checkpoint, device)
    checkpoint_guard = q_checkpoint.get("guard", {})
    minimum_probability_margin = (
        args.minimum_probability_margin
        if args.minimum_probability_margin is not None
        else float(checkpoint_guard.get("minimum_probability_margin", 0.05))
    )
    maximum_normalized_distance = (
        args.maximum_normalized_distance
        if args.maximum_normalized_distance is not None
        else float(checkpoint_guard.get("maximum_normalized_distance", 0.25))
    )
    authorized = bool(q_checkpoint.get("action_ranking_authorized", False))
    if not authorized and not args.allow_developmental:
        raise RuntimeError(
            "Q_ETSF validation gate did not authorize ranking; pass --allow-developmental "
            "only for a labeled simulator experiment"
        )
    unnorm_key = args.unnorm_key or f"{args.task}_1k"
    actor = (
        get_model(model_config(args.model_path, unnorm_key), torch_dtype=torch.bfloat16)
        .eval()
        .to(device)
    )
    capture = install_hidden_hook(actor)
    action_scale = reference_action_scale(args.baseline_rollouts, device)
    env = RoboTwinEnv(
        cfg=environment_config(args.robotwin_root, seeds_path, args.task, len(args.seeds)),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "collecting",
        "task": args.task,
        "body": BODY,
        "evaluation_seeds": args.seeds,
        "baseline_rollouts": str(args.baseline_rollouts),
        "q_checkpoint": str(args.q_checkpoint),
        "q_validation_authorized": authorized,
        "developmental_override": bool(args.allow_developmental and not authorized),
        "candidate_generator": {
            "count": 1 + len(args.blends),
            "blends": args.blends,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "preserve_grippers": True,
        },
        "guard": {
            "minimum_probability_margin": minimum_probability_margin,
            "maximum_normalized_distance": maximum_normalized_distance,
            "source": "command_line" if args.minimum_probability_margin is not None or args.maximum_normalized_distance is not None else "validation_selected_checkpoint",
        },
        "episodes": [],
    }
    try:
        for index, seed in enumerate(args.seeds):
            obs, _ = env.reset(env_seeds=[seed])
            generated = generate_candidates(
                actor,
                obs,
                capture,
                seed,
                args.blends,
                args.temperature,
                args.top_k,
                True,
                action_scale,
            )
            actions = generated.pop("candidate_actions_tensor")
            logits, probabilities = score_candidates(
                critic,
                generated["initial_hidden"].astype(np.float32),
                actions,
                generated["source_logprobs"],
            )
            proposed = int(np.argmax(probabilities))
            margin = float(probabilities[proposed] - probabilities[0])
            distance = float(generated["normalized_l2_from_baseline"][proposed])
            reasons = []
            selected = proposed
            if proposed != 0 and margin < minimum_probability_margin:
                reasons.append("insufficient_probability_margin")
                selected = 0
            if proposed != 0 and distance > maximum_normalized_distance:
                reasons.append("candidate_too_far_from_actor")
                selected = 0
            outcome = evaluate_first_chunk_branch(actor, env, seed, actions[selected])
            baseline = baseline_results[seed]
            item = {
                "index": index,
                "seed": seed,
                "baseline_success": baseline["success"],
                "baseline_steps": baseline["steps"],
                "proposed_candidate": proposed,
                "selected_candidate": selected,
                "fallback_reasons": reasons,
                "candidate_logits": logits.tolist(),
                "candidate_probabilities": probabilities.tolist(),
                "normalized_l2_from_baseline": generated["normalized_l2_from_baseline"].tolist(),
                "selected_success": outcome["success"],
                "selected_steps": outcome["steps"],
                "paired_success_difference": int(outcome["success"]) - int(baseline["success"]),
                "wall_seconds": outcome["wall_seconds"],
            }
            manifest["episodes"].append(item)
            atomic_json(args.output / "manifest.json", manifest)
            print("EVALUATED=" + json.dumps(item, sort_keys=True), flush=True)
    finally:
        env.venv.close(clear_cache=False)

    differences = np.asarray(
        [item["paired_success_difference"] for item in manifest["episodes"]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260826)
    bootstrap = differences[
        rng.integers(0, len(differences), size=(10000, len(differences)))
    ].mean(1)
    manifest["status"] = "complete"
    manifest["baseline_successes"] = int(
        sum(item["baseline_success"] for item in manifest["episodes"])
    )
    manifest["etsf_successes"] = int(
        sum(item["selected_success"] for item in manifest["episodes"])
    )
    manifest["baseline_success_rate"] = manifest["baseline_successes"] / len(args.seeds)
    manifest["etsf_success_rate"] = manifest["etsf_successes"] / len(args.seeds)
    manifest["paired_success_difference"] = float(differences.mean())
    manifest["paired_difference_ci95"] = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    manifest["changed_episodes"] = int(
        sum(item["selected_candidate"] != 0 for item in manifest["episodes"])
    )
    manifest["fallback_episodes"] = int(
        sum(bool(item["fallback_reasons"]) for item in manifest["episodes"])
    )
    atomic_json(args.output / "manifest.json", manifest)
    print(
        "EVALUATION_COMPLETE="
        + json.dumps(
            {
                key: manifest[key]
                for key in [
                    "baseline_successes",
                    "etsf_successes",
                    "baseline_success_rate",
                    "etsf_success_rate",
                    "paired_success_difference",
                    "paired_difference_ci95",
                    "changed_episodes",
                    "fallback_episodes",
                ]
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
