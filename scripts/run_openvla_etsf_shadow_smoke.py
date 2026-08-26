#!/usr/bin/env python3
"""Run a non-controlling ETSF shadow head on an OpenVLA-OFT RoboTwin rollout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def openvla_config(model_path: Path):
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "action_dim": 14,
            "num_action_chunks": 25,
            "add_value_head": False,
            "value_type": "action_level",
            "proprio_dim": 14,
            "use_proprio": True,
            "use_film": False,
            "max_prompt_length": 512,
            "unnorm_key": "move_can_pot_1k",
            "num_images_in_input": 1,
        }
    )


def robotwin_config(robotwin_root: Path, seeds_path: Path):
    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": True,
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": 200,
            "max_episode_steps": 200,
            "is_eval": True,
            "assets_path": str(robotwin_root),
            "seeds_path": str(seeds_path),
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/etsf_openvla_shadow_video",
            },
            "task_config": {
                "task_name": "move_can_pot",
                "step_lim": 200,
                "planner_backend": "mplib",
                "render_freq": 0,
                "episode_num": 100,
                "use_seed": False,
                "save_freq": 15,
                "embodiment": ["piper", "piper", 0.6],
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
                    "collect_wrist_camera": False,
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
                "save_path": "/tmp/etsf_openvla_shadow_data",
                "clear_cache_freq": 1,
                "collect_data": False,
                "eval_video_log": False,
            },
        }
    )


def install_hidden_hook(model):
    capture = {}
    original = model._discrete_prediction

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        capture["last_hidden_states"] = result[3].detach()
        return result

    model._discrete_prediction = wrapped
    return capture, original


def predict(model, obs):
    return model.predict_action_batch(
        env_obs=obs,
        do_sample=False,
        temperature=1.0,
        top_k=-1,
        calulate_values=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robotwin-code", type=Path, required=True)
    parser.add_argument("--stage3-code", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260826)
    np.random.seed(20260826)

    os.environ["ASSETS_PATH"] = str(args.robotwin_root)
    os.environ.setdefault(
        "VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    os.environ.setdefault(
        "VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    sys.path.insert(0, str(args.rlinf_root))
    sys.path.insert(0, str(args.robotwin_code))
    sys.path.insert(0, str(args.stage3_code))

    import run_stage3 as stage3
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.models.embodiment.openvla_oft.official import get_model

    device = torch.device("cuda:0")
    seeds_path = args.rlinf_root / "rlinf/envs/robotwin/seeds/eval_seeds.json"
    env = RoboTwinEnv(
        cfg=robotwin_config(args.robotwin_root, seeds_path),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    obs, _ = env.reset(env_seeds=[args.seed])

    model = get_model(openvla_config(args.model_path), torch_dtype=torch.bfloat16)
    model = model.eval().to(device)
    shadow = stage3.FactorizedEventTransport(4096).eval().to(device)
    # Deterministic behavior must remain identical after installing the hook.
    reference_actions, _ = predict(model, obs)
    capture, _ = install_hidden_hook(model)
    hooked_actions, _ = predict(model, obs)
    action_invariant = torch.equal(reference_actions, hooked_actions)
    if not action_invariant:
        raise RuntimeError("ETSF hidden hook changed deterministic OpenVLA actions")

    hidden_history = []
    dts = []
    rows = []
    success = False
    started = time.time()

    for chunk_index in range(8):
        if chunk_index == 0:
            actions = hooked_actions
        else:
            actions, _ = predict(model, obs)
        last_hidden = capture["last_hidden_states"]
        anchor = last_hidden[:, -model.action_dim * model.num_action_chunks - 1]
        hidden_history.append(anchor.float())
        dts.append(1.0 if chunk_index == 0 else float(model.num_action_chunks))
        inputs = torch.stack(hidden_history, dim=1)
        times = torch.tensor([dts], device=device)
        mask = torch.ones(1, len(hidden_history), dtype=torch.bool, device=device)
        with torch.inference_mode():
            shadow_output = shadow(inputs, times, mask, torch.zeros(1, device=device))
            semantic = stage3.decode_semantic(shadow_output["successor_logits"])
            durations = stage3.duration_statistics(shadow_output)
        goal = stage3.s2.MODEL_EVENTS.index("eK")
        shadow_goal = float(semantic[0, -1, goal])
        predicted_duration = float(durations[0, -1].mean())

        obs, reward, terminated, truncated, infos = env.step(
            actions, auto_reset=False
        )
        success_value = infos.get("success", [False])
        if isinstance(success_value, torch.Tensor):
            success = success or bool(success_value.flatten()[0])
        else:
            success = success or bool(np.asarray(success_value).reshape(-1)[0])
        rows.append(
            {
                "chunk": chunk_index,
                "hidden_norm": float(anchor.float().norm(dim=-1).mean()),
                "shadow_goal_value": shadow_goal,
                "shadow_mean_duration": predicted_duration,
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
                "reward": float(torch.as_tensor(reward).flatten()[0]),
                "success": success,
                "terminated": bool(torch.as_tensor(terminated).flatten()[0]),
                "truncated": bool(torch.as_tensor(truncated).flatten()[0]),
            }
        )
        if rows[-1]["terminated"] or rows[-1]["truncated"]:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(shadow.state_dict(), args.output / "shadow_random_init.pt")
    summary = {
        "status": "openvla_etsf_shadow_wiring_passed",
        "seed": args.seed,
        "task": "move_can_pot",
        "embodiment": "piper_piper_0.6",
        "action_invariant": action_invariant,
        "openvla_hidden_dim": 4096,
        "etsf_hidden_dim": stage3.SEMANTIC_HIDDEN,
        "chunks_executed": len(rows),
        "success": success,
        "wall_seconds": time.time() - started,
        "shadow_training": "random_init_wiring_only",
        "rows": rows,
    }
    (args.output / "shadow_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("OPENVLA_ETSF_SHADOW=" + json.dumps(summary, sort_keys=True), flush=True)
    env.venv.close(clear_cache=False)


if __name__ == "__main__":
    main()
