#!/usr/bin/env python3
"""Run the complete C+expert-root shared-head upgrade after the formal pipeline.

This CPU-only detached watcher deliberately waits for the existing formal
collection, C-only LOBO, paired-success study, and postformal ablation to finish
before reserving the single RTX 4090.  It then runs, in order:

1. the complete five-body 100-decision/400-branch scripted e3/e4 supplement;
2. its immutable five-body binding;
3. five strict C+supplement outer-LOBO folds;
4. the full 1,000-pair enhanced best-of-4 study; and
5. the full 1,000-pair enhanced best-of-8 postformal study.

No formal collector, formal branch root, C-only fold, or formal result is
modified.  Every enhanced evaluation explicitly requires the exact supplement
binding SHA-256, so a C-only or mixed-fold result cannot be mislabeled.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_robotwin2_postformal_shared_head_upgrade_watcher_v1"
UPSTREAM_FORMAT = "etsf_robotwin2_five_body_postformal_ablation_watcher_v1"
SUPPLEMENT_MANIFEST_FORMAT = "etsf_robotwin2_proper_world_supplement_manifest_v1"
SUPPLEMENT_BINDING_FORMAT = (
    "etsf_robotwin2_five_body_proper_world_supplement_binding_v1"
)
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
TARGET_EVENTS = ("e3", "e4")
SUPPLEMENT_HORIZONS = (10, 25, 50, 100, 200)
SUPPLEMENT_RESERVE_SEED_START = 2026081000
SUPPLEMENT_RESERVE_SEEDS_PER_SLOT = 16
SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE = 2026081800
EXPECTED_SUPPLEMENT_DECISIONS_PER_BODY = 20
EXPECTED_SUPPLEMENT_DECISIONS = 100
EXPECTED_SUPPLEMENT_BRANCHES = 400
EXPECTED_GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"
DEFAULT_ETSF_SITE = Path(
    "/home/user/anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages"
)


class SharedHeadUpgradeError(RuntimeError):
    """The upstream, supplement, training, or paired upgrade failed closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SharedHeadUpgradeError(f"{label} must be a real JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SharedHeadUpgradeError(f"{label} must be a JSON object")
    return value


def verify_logical_sha(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("logical_sha256", None)
    if declared != canonical_sha256(unsigned):
        raise SharedHeadUpgradeError(f"{label} logical SHA-256 mismatch")


def validate_upstream_state(value: Mapping[str, Any]) -> None:
    if value.get("format") != UPSTREAM_FORMAT or value.get("status") != "complete":
        raise SharedHeadUpgradeError("formal baseline/ablation pipeline is not complete")
    summary = Path(str(value.get("summary", "")))
    if (
        not summary.is_file()
        or summary.is_symlink()
        or sha256_file(summary) != value.get("summary_file_sha256")
    ):
        raise SharedHeadUpgradeError("formal postformal-ablation summary is missing/tampered")


def supplement_reserve_roster(body: str) -> list[dict[str, Any]]:
    if body not in BODIES:
        raise SharedHeadUpgradeError("unknown supplement body")
    rows = []
    body_index = BODIES.index(body)
    for condition_index, condition in enumerate(CONDITIONS):
        for slot, horizon in enumerate(SUPPLEMENT_HORIZONS):
            global_slot = (
                (body_index * len(CONDITIONS) + condition_index)
                * len(SUPPLEMENT_HORIZONS)
                + slot
            )
            first = (
                SUPPLEMENT_RESERVE_SEED_START
                + global_slot * SUPPLEMENT_RESERVE_SEEDS_PER_SLOT
            )
            rows.append(
                {
                    "slot_key": f"{condition}|horizon_slot={slot}",
                    "condition": condition,
                    "horizon_slot": slot,
                    "remaining_action_budget": horizon,
                    "ordered_requested_seeds": list(
                        range(first, first + SUPPLEMENT_RESERVE_SEEDS_PER_SLOT)
                    ),
                }
            )
    flattened = [seed for row in rows for seed in row["ordered_requested_seeds"]]
    if (
        len(set(flattened)) != len(flattened)
        or min(flattened) < SUPPLEMENT_RESERVE_SEED_START
        or max(flattened) >= SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE
    ):
        raise SharedHeadUpgradeError("supplement reserve roster changed")
    return rows


def supplement_manifest_complete(path: Path, body: str) -> bool:
    """Metadata-only resumability check; the materializer performs full validation."""

    try:
        value = read_json(path, f"{body} supplement manifest")
        verify_logical_sha(value, f"{body} supplement manifest")
    except (OSError, ValueError, json.JSONDecodeError, SharedHeadUpgradeError):
        return False
    groups = value.get("groups")
    attempts = value.get("attempts")
    selected = value.get("selected_seed_by_slot")
    roster = supplement_reserve_roster(body)
    if (
        value.get("format") != SUPPLEMENT_MANIFEST_FORMAT
        or value.get("body") != body
        or value.get("conditions") != list(CONDITIONS)
        or value.get("collection_status") != "complete"
        or value.get("reserve_roster") != roster
        or value.get("pre_registered_seeds")
        != [seed for row in roster for seed in row["ordered_requested_seeds"]]
        or not isinstance(groups, list)
        or len(groups) != EXPECTED_SUPPLEMENT_DECISIONS_PER_BODY
        or not isinstance(attempts, list)
        or not isinstance(selected, Mapping)
        or set(selected) != {row["slot_key"] for row in roster}
    ):
        return False
    attempt_by_id = {
        item.get("attempt_id"): item
        for item in attempts
        if isinstance(item, Mapping) and isinstance(item.get("attempt_id"), str)
    }
    if len(attempt_by_id) != len(attempts):
        return False
    design = {
        (
            item.get("condition"),
            item.get("horizon_slot"),
            item.get("requested_seed"),
            item.get("scripted_root_event"),
        )
        for item in groups
        if isinstance(item, Mapping)
    }
    expected = set()
    consumed_attempts = set()
    for row in roster:
        seed = selected.get(row["slot_key"])
        seeds = row["ordered_requested_seeds"]
        if isinstance(seed, bool) or seed not in seeds:
            return False
        selected_index = seeds.index(seed)
        for rejected_seed in seeds[:selected_index]:
            attempt_id = (
                f"{row['slot_key']}|requested_seed={rejected_seed}"
            )
            attempt = attempt_by_id.get(attempt_id)
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("status") != "rejected_before_actor_outcomes"
                or attempt.get(
                    "actor_candidate_outcomes_executed_before_selection"
                )
                is not False
            ):
                return False
            consumed_attempts.add(attempt_id)
        selected_attempt_id = f"{row['slot_key']}|requested_seed={seed}"
        selected_attempt = attempt_by_id.get(selected_attempt_id)
        if (
            not isinstance(selected_attempt, Mapping)
            or selected_attempt.get("status") != "complete"
            or selected_attempt.get("selected_before_actor_candidate_outcomes")
            is not True
            or selected_attempt.get(
                "actor_candidate_outcomes_executed_before_selection"
            )
            is not False
        ):
            return False
        consumed_attempts.add(selected_attempt_id)
        expected.update(
            (
                row["condition"],
                row["horizon_slot"],
                seed,
                event,
            )
            for event in TARGET_EVENTS
        )
    return design == expected and consumed_attempts == set(attempt_by_id)


def gpu_compute_pids(expected_uuid: str) -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SharedHeadUpgradeError("nvidia-smi failed while checking the RTX 4090")
    pids = []
    for raw in result.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",")]
        if len(fields) == 2 and fields[0] == expected_uuid and fields[1].isdigit():
            pids.append(int(fields[1]))
    return sorted(set(pids))


def runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": ":".join(
                (
                    str(args.code_root),
                    str(args.lerobot_root / "src"),
                    str(args.lerobot_site),
                    str(args.robotwin_eval_site),
                    str(args.etsf_site),
                    str(args.robotwin_root),
                    str(args.robotwin_root / "envs/curobo/src"),
                )
            ),
            "ASSETS_PATH": str(args.robotwin_root),
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def supplement_collector_command(args: argparse.Namespace, body: str) -> list[str]:
    return [
        str(args.robotwin_python),
        str(args.code_root / "collect_robotwin2_scripted_expert_root_actor_branches_v1.py"),
        "--body",
        body,
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--actor-authority",
        str(args.actor_authority),
        "--vlm-metadata-path",
        str(args.vlm_metadata),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--output",
        str(args.supplement_root / body),
        "--conditions",
        *CONDITIONS,
        "--action-exec-steps",
        "5",
    ]


def materializer_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.training_python),
        str(args.code_root / "materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py"),
        "--primary-binding",
        str(args.primary_binding),
        "--actor-authority",
        str(args.actor_authority),
    ]
    for body in BODIES:
        command.extend(
            (
                "--body-manifest",
                f"{body}={args.supplement_root / body / 'manifest.json'}",
            )
        )
    command.extend(("--output", str(args.supplement_binding)))
    return command


def lobo_command(args: argparse.Namespace, supplement_sha256: str) -> list[str]:
    return [
        str(args.system_python),
        str(args.code_root / "watch_robotwin2_five_body_branches_to_lobo_training_v1.py"),
        "--branches-root",
        str(args.primary_branches_root),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--materialization-receipt",
        str(args.materialization_receipt),
        "--actor-authority",
        str(args.actor_authority),
        "--binding",
        str(args.primary_binding),
        "--supplement-binding",
        str(args.supplement_binding),
        "--supplement-binding-sha256",
        supplement_sha256,
        "--output-root",
        str(args.augmented_lobo_root),
        "--state",
        str(args.augmented_lobo_state),
        "--run-exit",
        str(args.augmented_lobo_run_exit),
        "--trainer",
        str(args.code_root / "train_robotwin2_five_body_lobo_shared_event_head_v1.py"),
        "--training-python",
        str(args.training_python),
        "--poll-seconds",
        str(args.poll_seconds),
        "--expected-gpu-uuid",
        args.expected_gpu_uuid,
    ]


def fold_arguments(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for body in BODIES:
        result.extend(
            (
                "--lobo-fold",
                f"{body}={args.augmented_lobo_root / f'outer_lobo_{body}'}",
            )
        )
    return result


def paired_n4_command(args: argparse.Namespace, supplement_sha256: str) -> list[str]:
    return [
        str(args.robotwin_python),
        str(args.code_root / "run_robotwin2_five_body_paired_success_v1.py"),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--vlm-metadata-path",
        str(args.vlm_metadata),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--preregistration",
        str(args.metrics_preregistration),
        *fold_arguments(args),
        "--required-supplement-binding-sha256",
        supplement_sha256,
        "--output",
        str(args.augmented_n4_root),
        "--action-exec-steps",
        "5",
        "--max-steps",
        "200",
        "--fps",
        "15.0",
    ]


def paired_n8_command(args: argparse.Namespace, supplement_sha256: str) -> list[str]:
    return [
        str(args.robotwin_python),
        str(args.code_root / "run_robotwin2_five_body_postformal_candidate_pool_v1.py"),
        "--actor-checkpoint",
        str(args.actor_checkpoint),
        "--vlm-metadata-path",
        str(args.vlm_metadata),
        "--robotwin-root",
        str(args.robotwin_root),
        "--event-spec",
        str(args.event_spec),
        "--reference-preregistration",
        str(args.metrics_preregistration),
        *fold_arguments(args),
        "--required-supplement-binding-sha256",
        supplement_sha256,
        "--candidate-count",
        "8",
        "--output",
        str(args.augmented_n8_root),
        "--action-exec-steps",
        "5",
        "--max-steps",
        "200",
        "--fps",
        "15.0",
    ]


def evaluator_command(args: argparse.Namespace) -> list[str]:
    outcomes = args.augmented_n4_root / "paired_outcomes.json"
    return [
        str(args.system_python),
        str(args.code_root / "evaluate_robotwin2_cross_embodiment_paired_success_v1.py"),
        "--input",
        str(outcomes),
        "--input-file-sha256",
        sha256_file(outcomes),
        "--output",
        str(args.augmented_n4_root / "paired_success_report.json"),
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-state", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--primary-branches-root", type=Path, required=True)
    parser.add_argument("--primary-binding", type=Path, required=True)
    parser.add_argument("--actor-authority", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--vlm-metadata", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--metrics-preregistration", type=Path, required=True)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--supplement-binding", type=Path, required=True)
    parser.add_argument("--augmented-lobo-root", type=Path, required=True)
    parser.add_argument("--augmented-lobo-state", type=Path, required=True)
    parser.add_argument("--augmented-lobo-run-exit", type=Path, required=True)
    parser.add_argument("--augmented-n4-root", type=Path, required=True)
    parser.add_argument("--augmented-n8-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument(
        "--robotwin-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/RoboTwin2/bin/python"),
    )
    parser.add_argument(
        "--training-python",
        type=Path,
        default=Path("/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python"),
    )
    parser.add_argument("--system-python", type=Path, default=Path("/usr/bin/python3"))
    parser.add_argument(
        "--lerobot-root", type=Path, default=Path("/home/user/etsf_stage0/lerobot")
    )
    parser.add_argument(
        "--lerobot-site",
        type=Path,
        default=Path(
            "/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/"
            "lib/python3.10/site-packages"
        ),
    )
    parser.add_argument(
        "--robotwin-eval-site",
        type=Path,
        default=Path(
            "/home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/"
            "lib/python3.10/site-packages"
        ),
    )
    parser.add_argument("--etsf-site", type=Path, default=DEFAULT_ETSF_SITE)
    parser.add_argument("--expected-gpu-uuid", default=EXPECTED_GPU_UUID)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    path_names = (
        "upstream_state",
        "code_root",
        "primary_branches_root",
        "primary_binding",
        "actor_authority",
        "actor_checkpoint",
        "materialization_receipt",
        "robotwin_root",
        "vlm_metadata",
        "event_spec",
        "metrics_preregistration",
        "supplement_root",
        "supplement_binding",
        "augmented_lobo_root",
        "augmented_lobo_state",
        "augmented_lobo_run_exit",
        "augmented_n4_root",
        "augmented_n8_root",
        "state",
        "run_exit",
        "lock",
        "log_root",
        "robotwin_python",
        "training_python",
        "system_python",
        "lerobot_root",
        "lerobot_site",
        "robotwin_eval_site",
        "etsf_site",
    )
    for name in path_names:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.poll_seconds <= 0:
        raise SharedHeadUpgradeError("poll interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = normalize_args(parse_args(argv))
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = args.lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SharedHeadUpgradeError("another shared-head upgrade watcher is active") from error

    def write_state(status: str, **extra: Any) -> None:
        atomic_json(
            args.state,
            {
                "format": FORMAT,
                "status": status,
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "upstream_state": str(args.upstream_state),
                "supplement_root": str(args.supplement_root),
                "supplement_binding": str(args.supplement_binding),
                "augmented_lobo_root": str(args.augmented_lobo_root),
                "augmented_n4_root": str(args.augmented_n4_root),
                "augmented_n8_root": str(args.augmented_n8_root),
                "expected_supplement_decisions": EXPECTED_SUPPLEMENT_DECISIONS,
                "expected_supplement_branches": EXPECTED_SUPPLEMENT_BRANCHES,
                "gpu_uuid": args.expected_gpu_uuid,
                **extra,
            },
        )

    def wait_idle(status: str) -> None:
        while True:
            pids = gpu_compute_pids(args.expected_gpu_uuid)
            if not pids:
                return
            write_state(status, active_compute_pids=pids, gpu_reserved_by_watcher=False)
            time.sleep(args.poll_seconds)

    def run_stage(
        stage: str,
        command: Sequence[str],
        log_name: str,
        *,
        cwd: Path,
        environment: Mapping[str, str],
    ) -> None:
        log_path = args.log_root / log_name
        write_state(
            stage,
            command=list(command),
            log=str(log_path),
            gpu_reserved_by_watcher=True,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("\nUPGRADE_INVOCATION=" + json.dumps(list(command)) + "\n")
            stream.flush()
            result = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise SharedHeadUpgradeError(f"{stage} exited {result.returncode}")

    while True:
        try:
            upstream = read_json(args.upstream_state, "upstream ablation state")
        except (OSError, ValueError, json.JSONDecodeError, SharedHeadUpgradeError):
            upstream = {}
        if upstream.get("status") == "failed":
            raise SharedHeadUpgradeError("formal upstream pipeline failed")
        if upstream.get("status") == "complete":
            validate_upstream_state(upstream)
            break
        write_state(
            "waiting_for_complete_formal_c_only_pipeline",
            upstream_status=upstream.get("status"),
            gpu_reserved_by_watcher=False,
        )
        time.sleep(args.poll_seconds)

    static_paths = (
        args.code_root,
        args.primary_branches_root,
        args.primary_binding,
        args.actor_authority,
        args.actor_checkpoint,
        args.materialization_receipt,
        args.robotwin_root,
        args.vlm_metadata,
        args.event_spec,
        args.metrics_preregistration,
        args.robotwin_python,
        args.training_python,
        args.system_python,
        args.lerobot_root,
        args.lerobot_site,
        args.robotwin_eval_site,
        args.etsf_site,
    )
    if any(not path.exists() or path.is_symlink() for path in static_paths):
        raise SharedHeadUpgradeError("one or more immutable upgrade inputs are missing/symbolic")
    required_scripts = (
        "collect_robotwin2_scripted_expert_root_actor_branches_v1.py",
        "materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py",
        "train_robotwin2_five_body_lobo_shared_event_head_v1.py",
        "watch_robotwin2_five_body_branches_to_lobo_training_v1.py",
        "run_robotwin2_five_body_paired_success_v1.py",
        "run_robotwin2_five_body_postformal_candidate_pool_v1.py",
        "evaluate_robotwin2_cross_embodiment_paired_success_v1.py",
    )
    if any(
        not (args.code_root / name).is_file()
        or (args.code_root / name).is_symlink()
        for name in required_scripts
    ):
        raise SharedHeadUpgradeError("deployed upgrade code directory is incomplete")
    environment = runtime_environment(args)

    completed_bodies = []
    for body in BODIES:
        manifest_path = args.supplement_root / body / "manifest.json"
        if not supplement_manifest_complete(manifest_path, body):
            wait_idle("waiting_for_idle_rtx4090_before_supplement_collection")
            run_stage(
                f"collecting_scripted_expert_root_supplement_{body}",
                supplement_collector_command(args, body),
                f"collect_{body}.log",
                cwd=args.robotwin_root,
                environment=environment,
            )
        if not supplement_manifest_complete(manifest_path, body):
            raise SharedHeadUpgradeError(f"{body} supplement did not complete 20 decisions")
        completed_bodies.append(body)
        write_state(
            "supplement_body_complete",
            completed_supplement_bodies=list(completed_bodies),
            gpu_reserved_by_watcher=False,
        )

    run_stage(
        "materializing_exact_five_body_supplement_binding",
        materializer_command(args),
        "materialize_supplement_binding.log",
        cwd=args.code_root,
        environment=environment,
    )
    binding = read_json(args.supplement_binding, "supplement binding")
    verify_logical_sha(binding, "supplement binding")
    if (
        binding.get("format") != SUPPLEMENT_BINDING_FORMAT
        or binding.get("materializer_provenance", {}).get("complete_decisions")
        != EXPECTED_SUPPLEMENT_DECISIONS
        or binding.get("materializer_provenance", {}).get("complete_branches")
        != EXPECTED_SUPPLEMENT_BRANCHES
    ):
        raise SharedHeadUpgradeError("supplement binding is not the complete 100/400 design")
    supplement_sha256 = sha256_file(args.supplement_binding)

    run_stage(
        "running_five_fold_c_plus_supplement_lobo",
        lobo_command(args, supplement_sha256),
        "augmented_lobo_watcher.log",
        cwd=args.code_root,
        environment=environment,
    )
    if args.augmented_lobo_run_exit.read_text(encoding="utf-8").strip() != "0":
        raise SharedHeadUpgradeError("augmented LOBO did not finish successfully")

    wait_idle("waiting_for_idle_rtx4090_before_augmented_n4")
    run_stage(
        "running_complete_augmented_best_of_4_paired_evaluation",
        paired_n4_command(args, supplement_sha256),
        "augmented_n4_paired.log",
        cwd=args.robotwin_root,
        environment=environment,
    )
    n4_outcomes = args.augmented_n4_root / "paired_outcomes.json"
    n4_report = args.augmented_n4_root / "paired_success_report.json"
    if not n4_outcomes.is_file():
        raise SharedHeadUpgradeError("augmented N=4 runner did not produce outcomes")
    if not n4_report.exists():
        run_stage(
            "evaluating_complete_augmented_best_of_4_outcomes",
            evaluator_command(args),
            "augmented_n4_evaluator.log",
            cwd=args.code_root,
            environment=os.environ.copy(),
        )
    if not n4_report.is_file() or n4_report.is_symlink():
        raise SharedHeadUpgradeError("augmented N=4 evaluator did not produce a report")

    wait_idle("waiting_for_idle_rtx4090_before_augmented_n8")
    run_stage(
        "running_complete_augmented_best_of_8_paired_evaluation",
        paired_n8_command(args, supplement_sha256),
        "augmented_n8_paired.log",
        cwd=args.robotwin_root,
        environment=environment,
    )
    n8_receipt = args.augmented_n8_root / "completion_receipt.json"
    if not n8_receipt.is_file():
        raise SharedHeadUpgradeError("augmented N=8 runner did not complete 1000 pairs")
    n8_report = args.augmented_n8_root / "paired_candidate_pool_report.json"
    if not n8_report.is_file() or n8_report.is_symlink():
        raise SharedHeadUpgradeError("augmented N=8 runner did not produce a report")

    atomic_text(args.run_exit, "0\n")
    write_state(
        "complete",
        supplement_binding_sha256=supplement_sha256,
        augmented_lobo_summary=str(
            args.augmented_lobo_root / "five_fold_training_summary.json"
        ),
        augmented_n4_report=str(n4_report),
        augmented_n4_report_file_sha256=sha256_file(n4_report),
        augmented_n8_report=str(n8_report),
        augmented_n8_report_file_sha256=sha256_file(n8_report),
        completed_pairs_by_candidate_count={"4": 1000, "8": 1000},
        completed_rollouts_by_candidate_count={"4": 2000, "8": 2000},
        gpu_reserved_by_watcher=False,
    )
    lock_stream.flush()
    return 0


def record_failure(
    state: Path | None, run_exit: Path | None, error: BaseException
) -> None:
    if run_exit is not None:
        atomic_text(run_exit, "1\n")
    if state is not None:
        atomic_json(
            state,
            {
                "format": FORMAT,
                "status": "failed",
                "updated_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )


if __name__ == "__main__":
    state_path: Path | None = None
    run_exit_path: Path | None = None
    try:
        parsed = normalize_args(parse_args())
        state_path = parsed.state
        run_exit_path = parsed.run_exit
        raise SystemExit(main(sys.argv[1:]))
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            record_failure(state_path, run_exit_path, error)
        raise


__all__ = [
    "SharedHeadUpgradeError",
    "evaluator_command",
    "fold_arguments",
    "lobo_command",
    "materializer_command",
    "paired_n4_command",
    "paired_n8_command",
    "supplement_collector_command",
    "supplement_manifest_complete",
    "supplement_reserve_roster",
    "validate_upstream_state",
]
