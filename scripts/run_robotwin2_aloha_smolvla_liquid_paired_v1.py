#!/usr/bin/env python3
"""Paired clean ``move_can_pot`` comparison for official SmolVLA and v14.

The actor, analytic single-arm adapter, initial seeds and flow-noise schedule
are shared.  The control always executes candidate zero; the treatment changes
only the candidate index selected by the frozen liquid shared head.  Small
index batches are intentionally separate processes so SAPIEN/MPLib state never
accumulates over the full 100-pair study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector
import robotwin2_liquid_shared_event_runtime_v1 as liquid_runtime


FORMAT = "etsf_robotwin2_aloha_official_smolvla_liquid_paired_v1"
PAIR_FORMAT = "etsf_robotwin2_aloha_official_smolvla_liquid_pair_v1"
REPORT_FORMAT = "etsf_robotwin2_aloha_official_smolvla_liquid_paired_report_v1"
METHOD_BASELINE = "official_smolvla_candidate0"
METHOD_LIQUID = "official_smolvla_liquid_best_of_4"
BODY = "aloha-agilex"
CONDITION = "clean"


class PairedEvaluationError(RuntimeError):
    """A frozen actor, paired reset, rollout or report invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_seed_roster(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[int]]:
    path = path.expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise PairedEvaluationError("paired seed roster is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    seeds = value.get("seeds")
    if (
        value.get("format") != "etsf_robotwin2_aloha_move_can_pot_clean100_seed_roster_v1"
        or value.get("task") != collector.TASK
        or value.get("body") != BODY
        or value.get("condition") != CONDITION
        or not isinstance(seeds, list)
        or len(seeds) != 100
        or len(set(seeds)) != 100
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise PairedEvaluationError("paired seed roster contract is invalid")
    return value, [int(seed) for seed in seeds]


def reset_identity(task: Any, names: Sequence[str], objects: Sequence[Any]) -> str:
    value = {
        "object_names": list(names),
        "object_poses_sha256": array_sha256(collector.read_poses(objects)),
        "joint14_sha256": array_sha256(collector.current_aloha_joint_action14(task)),
        "ee16_sha256": array_sha256(collector.current_ee_action16(task)),
        "arm_tag": str(getattr(task, "arm_tag", "")),
        "height_reference_z": collector.success_height_reference_z(task),
    }
    return canonical_sha256(value)


def _append_observation(
    task: Any,
    objects: Sequence[Any],
    trajectory: list[np.ndarray],
    sim_times: list[float],
    ee_actions: list[np.ndarray],
) -> None:
    collector._append_physical_observation(task, objects, trajectory, sim_times)
    ee_actions.append(collector.current_ee_action16(task))


def run_rollout(
    *,
    method: str,
    seed: int,
    task_class: Any,
    task_args: Mapping[str, Any],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    joint_to_ee: Any,
    models: Sequence[Any],
    calibration: Mapping[str, Any],
    required_pose_names: set[str],
    protocol: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if method not in {METHOD_BASELINE, METHOD_LIQUID}:
        raise PairedEvaluationError(f"unknown method: {method}")
    max_steps = int(protocol["max_steps"])
    stride = int(protocol["stride"])
    fps = float(protocol["fps"])
    task = collector._new_task(
        task_class,
        {**dict(task_args), "step_lim": max_steps},
        seed,
        collector.DEFAULT_INSTRUCTION,
    )
    try:
        names, objects = collector.discover_pose_objects(task, required_pose_names)
        trajectory = [collector.read_poses(objects)]
        sim_times = [collector._sim_time(task)]
        ee_actions = [collector.current_ee_action16(task)]
        reset_sha = reset_identity(task, names, objects)
        height_reference_z = collector.success_height_reference_z(task)
        decisions = []
        query_index = 0
        while not collector._episode_done(task, max_steps):
            task.scene.step()
            _append_observation(task, objects, trajectory, sim_times, ee_actions)
            batch = collector.generate_candidate_batch(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=task,
                instruction=collector.DEFAULT_INSTRUCTION,
                scene_seed=seed,
                query_index=query_index,
                candidate_count=collector.CANDIDATE_COUNT,
                device=device,
                actor_action_contract="aloha_joint14",
                joint_to_ee=joint_to_ee,
            )
            remaining = max_steps - int(getattr(task, "take_action_cnt", 0))
            if method == METHOD_BASELINE:
                selected = 0
                scoring = None
            else:
                history = liquid_runtime.canonical_history_at_runtime(
                    trajectory=np.stack(trajectory),
                    sim_times=np.asarray(sim_times, dtype=np.float64),
                    ee_actions=np.asarray(ee_actions, dtype=np.float32),
                    names=names,
                    calibration=calibration,
                    success_height_reference_z=height_reference_z,
                    history_length=liquid_runtime.HISTORY_LENGTH,
                )
                score_batch = liquid_runtime.scoring_batch(
                    canonical_history=history,
                    native_candidates=batch.canonical_ee_actions,
                    remaining_action_budget=remaining,
                    action_exec_steps=stride,
                    fps=fps,
                    device=device,
                )
                scoring = liquid_runtime.score_candidates(models, score_batch)
                selected = int(scoring["selected_candidate_index"])
            if not 0 <= selected < collector.CANDIDATE_COUNT:
                raise PairedEvaluationError("selected candidate index is invalid")
            decisions.append(
                {
                    "query_index": query_index,
                    "remaining_action_budget": remaining,
                    "selected_candidate_index": selected,
                    "candidate0_actor_sha256": array_sha256(
                        batch.actor_actions[0]
                    ),
                    "candidate_pool_actor_sha256": array_sha256(
                        batch.actor_actions
                    ),
                    "candidate_pool_canonical_ee_sha256": array_sha256(
                        batch.canonical_ee_actions
                    ),
                    "active_arm": batch.active_arm,
                    "scoring": scoring,
                }
            )
            for action in batch.execution_actions[selected, :stride]:
                if collector._episode_done(task, max_steps):
                    break
                task.take_action(action, action_type=batch.execution_action_type)
                _append_observation(
                    task, objects, trajectory, sim_times, ee_actions
                )
            query_index += 1
        success = bool(getattr(task, "eval_success", False)) or bool(
            task.check_success()
        )
        predicates, events = collector.derive_predicates_and_events(
            np.stack(trajectory),
            np.asarray(sim_times, dtype=np.float64),
            names,
            success,
            calibration,
            height_reference_z,
        )
        del predicates
        max_event = int(np.max(events))
        stage_progress = 1.0 if success else max_event / 4.0
        return {
            "method": method,
            "seed": seed,
            "reset_sha256": reset_sha,
            "success": success,
            "terminal_max_event_id": max_event,
            "terminal_stage_progress": stage_progress,
            "action_steps": int(getattr(task, "take_action_cnt", 0)),
            "query_count": len(decisions),
            "initial_candidate0_actor_sha256": decisions[0][
                "candidate0_actor_sha256"
            ],
            "initial_candidate_pool_actor_sha256": decisions[0][
                "candidate_pool_actor_sha256"
            ],
            "decisions": decisions,
        }
    finally:
        task.close_env(clear_cache=False)


def load_actor_and_adapter(args: argparse.Namespace, device: torch.device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from robotwin2_frozen_smolvla_aloha_to_ee_policy_v1 import AlohaJointToWorldEE

    checkpoint = args.actor_checkpoint.expanduser().resolve()
    vlm = args.vlm_metadata_path.expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = str(device)
    config.vlm_model_name = str(vlm)
    config.load_vlm_weights = False
    if (
        config.action_feature is None
        or int(config.action_feature.shape[0]) != collector.ALOHA_JOINT_DIM
        or int(config.chunk_size) != 50
    ):
        raise PairedEvaluationError("official SmolVLA actor ABI changed")
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint, config=config, local_files_only=True, strict=True
    ).eval().to(device)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": str(vlm)},
        },
    )
    adapter = AlohaJointToWorldEE(
        args.robotwin_root
        / "assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf",
        args.robotwin_root / "assets/embodiments/aloha-agilex/config.yml",
    )
    return policy, preprocessor, postprocessor, adapter


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def mcnemar_exact(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    improved = int(np.sum((left == 0) & (right == 1)))
    regressed = int(np.sum((left == 1) & (right == 0)))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(improved, regressed) + 1)
        ) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_fail_liquid_success": improved,
        "baseline_success_liquid_fail": regressed,
        "discordant_pairs": discordant,
        "two_sided_exact_p_value": p_value,
    }


def finalize_report(
    *, output: Path, experiment: Mapping[str, Any], seeds: Sequence[int]
) -> dict[str, Any]:
    pairs = []
    for seed in seeds:
        path = output / "pairs" / f"seed_{seed}.json"
        if not path.is_file():
            raise PairedEvaluationError(f"paired result is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("format") != PAIR_FORMAT
            or value.get("experiment_logical_sha256")
            != experiment["logical_sha256"]
            or value.get("seed") != seed
            or value.get("baseline", {}).get("reset_sha256")
            != value.get("liquid", {}).get("reset_sha256")
            or value.get("baseline", {}).get("initial_candidate_pool_actor_sha256")
            != value.get("liquid", {}).get("initial_candidate_pool_actor_sha256")
        ):
            raise PairedEvaluationError(f"paired identity changed: {path}")
        pairs.append(value)
    baseline = np.asarray(
        [int(pair["baseline"]["success"]) for pair in pairs], dtype=np.int64
    )
    liquid = np.asarray(
        [int(pair["liquid"]["success"]) for pair in pairs], dtype=np.int64
    )
    baseline_stage = np.asarray(
        [pair["baseline"]["terminal_stage_progress"] for pair in pairs],
        dtype=np.float64,
    )
    liquid_stage = np.asarray(
        [pair["liquid"]["terminal_stage_progress"] for pair in pairs],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260901)
    indices = rng.integers(0, len(pairs), size=(20_000, len(pairs)))
    delta_samples = (liquid - baseline)[indices].mean(axis=1)
    stage_samples = (liquid_stage - baseline_stage)[indices].mean(axis=1)
    selected = [
        int(decision["selected_candidate_index"])
        for pair in pairs
        for decision in pair["liquid"]["decisions"]
    ]
    report = {
        "format": REPORT_FORMAT,
        "status": "complete_paired_clean100",
        "experiment_logical_sha256": experiment["logical_sha256"],
        "task": collector.TASK,
        "body": BODY,
        "condition": CONDITION,
        "pair_count": len(pairs),
        "baseline": {
            "method": METHOD_BASELINE,
            "successes": int(baseline.sum()),
            "success_rate": float(baseline.mean()),
            "wilson_95_interval": wilson(int(baseline.sum()), len(pairs)),
            "mean_stage_progress": float(baseline_stage.mean()),
        },
        "liquid": {
            "method": METHOD_LIQUID,
            "successes": int(liquid.sum()),
            "success_rate": float(liquid.mean()),
            "wilson_95_interval": wilson(int(liquid.sum()), len(pairs)),
            "mean_stage_progress": float(liquid_stage.mean()),
            "nonzero_candidate_selection_rate": float(
                np.mean(np.asarray(selected) != 0)
            ),
            "candidate_selection_counts": {
                str(index): selected.count(index)
                for index in range(collector.CANDIDATE_COUNT)
            },
        },
        "paired_delta": {
            "success_rate_liquid_minus_baseline": float(
                (liquid - baseline).mean()
            ),
            "success_rate_percentile_bootstrap_95_interval": np.quantile(
                delta_samples, [0.025, 0.975]
            ).tolist(),
            "stage_progress_liquid_minus_baseline": float(
                (liquid_stage - baseline_stage).mean()
            ),
            "stage_progress_percentile_bootstrap_95_interval": np.quantile(
                stage_samples, [0.025, 0.975]
            ).tolist(),
            "mcnemar": mcnemar_exact(baseline, liquid),
        },
        "actor_weights_updated": False,
        "only_intervention": "candidate_index_selection",
        "same_seed_and_initial_candidate_pool_verified": True,
        "pairs": [str(output / "pairs" / f"seed_{seed}.json") for seed in seeds],
    }
    atomic_json(output / "paired_report.json", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise PairedEvaluationError("formal paired comparison requires the RTX 4090")
    roster, seeds = load_seed_roster(args.seed_roster, args.seed_roster_sha256)
    protocol = collector.actor_execution.load_execution_protocol_file(
        args.actor_execution_protocol,
        args.actor_execution_protocol_sha256,
        expected_stride=50,
    )
    if protocol["task"] != collector.TASK:
        raise PairedEvaluationError("paired actor protocol task changed")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    experiment_path = output / "experiment.json"
    unsigned_experiment = {
        "format": FORMAT,
        "task": collector.TASK,
        "body": BODY,
        "condition": CONDITION,
        "actor_checkpoint_identity": collector.actor_checkpoint_identity(
            args.actor_checkpoint
        ),
        "liquid_training_summary": str(args.training_summary.expanduser().resolve()),
        "liquid_training_summary_sha256": args.training_summary_sha256,
        "seed_roster": str(args.seed_roster.expanduser().resolve()),
        "seed_roster_sha256": args.seed_roster_sha256,
        "seed_roster_source": roster.get("source"),
        "actor_execution_protocol": protocol,
        "actor_execution_protocol_file_sha256": args.actor_execution_protocol_sha256,
        "candidate_count": collector.CANDIDATE_COUNT,
        "actor_action_contract": "aloha_joint14",
        "single_arm_adapter_shared_by_both_methods": True,
        "actor_weights_updated": False,
        "methods": [METHOD_BASELINE, METHOD_LIQUID],
    }
    experiment = {
        **unsigned_experiment,
        "logical_sha256": canonical_sha256(unsigned_experiment),
    }
    if experiment_path.is_file():
        if json.loads(experiment_path.read_text(encoding="utf-8")) != experiment:
            raise PairedEvaluationError("existing paired experiment binding changed")
    else:
        atomic_json(experiment_path, experiment)
    if args.finalize_only:
        return finalize_report(output=output, experiment=experiment, seeds=seeds)

    start = args.seed_index_start
    stop = min(len(seeds), start + args.seed_index_count)
    if not 0 <= start < stop <= len(seeds):
        raise PairedEvaluationError("paired seed index slice is invalid")
    device = torch.device("cuda:0")
    policy, preprocessor, postprocessor, joint_to_ee = load_actor_and_adapter(
        args, device
    )
    models = liquid_runtime.load_frozen_ensemble(
        args.training_summary,
        args.training_summary_sha256,
        target_body=BODY,
        device=device,
    )
    event_spec, calibration = collector.analytic_event.load_event_spec(
        args.event_spec
    )
    del event_spec
    required_pose_names = set(collector.analytic_event.REQUIRED_OBJECTS)
    os.environ["ASSETS_PATH"] = str(args.robotwin_root.expanduser().resolve())
    module = __import__(f"envs.{collector.TASK}", fromlist=[collector.TASK])
    task_class = getattr(module, collector.TASK)
    task_args = collector._load_task_args(
        args.robotwin_root.expanduser().resolve(), BODY, CONDITION
    )
    for seed in seeds[start:stop]:
        pair_path = output / "pairs" / f"seed_{seed}.json"
        if pair_path.is_file():
            observed = json.loads(pair_path.read_text(encoding="utf-8"))
            if (
                observed.get("format") != PAIR_FORMAT
                or observed.get("seed") != seed
                or observed.get("experiment_logical_sha256")
                != experiment["logical_sha256"]
            ):
                raise PairedEvaluationError("existing paired seed result changed")
            continue
        baseline = run_rollout(
            method=METHOD_BASELINE,
            seed=seed,
            task_class=task_class,
            task_args=task_args,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            joint_to_ee=joint_to_ee,
            models=models,
            calibration=calibration,
            required_pose_names=required_pose_names,
            protocol=protocol,
            device=device,
        )
        liquid = run_rollout(
            method=METHOD_LIQUID,
            seed=seed,
            task_class=task_class,
            task_args=task_args,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            joint_to_ee=joint_to_ee,
            models=models,
            calibration=calibration,
            required_pose_names=required_pose_names,
            protocol=protocol,
            device=device,
        )
        if (
            baseline["reset_sha256"] != liquid["reset_sha256"]
            or baseline["initial_candidate_pool_actor_sha256"]
            != liquid["initial_candidate_pool_actor_sha256"]
        ):
            raise PairedEvaluationError(
                "paired methods did not share reset and initial candidate pool"
            )
        atomic_json(
            pair_path,
            {
                "format": PAIR_FORMAT,
                "experiment_logical_sha256": experiment["logical_sha256"],
                "seed": seed,
                "baseline": baseline,
                "liquid": liquid,
            },
        )
        print(
            "PAIR="
            + json.dumps(
                {
                    "seed": seed,
                    "baseline_success": baseline["success"],
                    "liquid_success": liquid["success"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol-sha256", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-summary-sha256", required=True)
    parser.add_argument("--seed-roster", type=Path, required=True)
    parser.add_argument("--seed-roster-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-index-start", type=int, default=0)
    parser.add_argument("--seed-index-count", type=int, default=4)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seed_index_start < 0 or args.seed_index_count <= 0:
        parser.error("invalid paired seed index slice")
    return args


def main() -> None:
    result = run(parse_args())
    if result is not None:
        print("PAIRED_REPORT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
