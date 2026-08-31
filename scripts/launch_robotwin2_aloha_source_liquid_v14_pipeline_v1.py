#!/usr/bin/env python3
"""Collect Aloha liquid histories, then train the source-only v14 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


FORMAT = "etsf_robotwin2_aloha_source_liquid_v14_pipeline_v1"
SOURCE_BODY = "aloha-agilex"
SEALED_TARGET_BODIES = ("arx-x5", "franka", "piper", "ur5")


class PipelineError(RuntimeError):
    """The remote source-only pipeline failed closed."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_state(path: Path, status: str, **extra: Any) -> None:
    atomic_json(
        path,
        {
            "format": FORMAT,
            "status": status,
            "updated_at_utc": now_utc(),
            "source_body": SOURCE_BODY,
            "sealed_target_bodies": list(SEALED_TARGET_BODIES),
            "target_payloads_passed_to_training": False,
            **extra,
        },
    )


def collector_environment(
    *, code_root: Path, path_root: Path, robotwin_root: Path
) -> dict[str, str]:
    """Reproduce the frozen SmolVLA+RoboTwin runtime import contract."""

    lerobot_root = path_root / "etsf_stage0/lerobot"
    lerobot_site = path_root / (
        "etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages"
    )
    etsf_site = path_root / (
        "anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages"
    )
    python_paths = (
        code_root / "scripts",
        lerobot_root / "src",
        lerobot_site,
        robotwin_root,
        robotwin_root / "envs/curobo/src",
        etsf_site,
    )
    if any(not path.exists() for path in python_paths):
        missing = [str(path) for path in python_paths if not path.exists()]
        raise PipelineError(f"collector runtime paths are missing: {missing}")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ":".join(str(path) for path in python_paths),
            "ASSETS_PATH": str(robotwin_root),
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def run(args: argparse.Namespace) -> None:
    code_root = args.code_root.expanduser().resolve()
    path_root = args.path_root.expanduser().resolve()
    robotwin_root = args.robotwin_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    state = output_root / "pipeline_state.json"
    source_root = output_root / "aloha_source_branches"
    manifest = source_root / "manifest.json"
    training_root = output_root / "liquid_v14_training"
    paired_root = output_root / "paired_clean100_official_smolvla_vs_liquid"
    collector = code_root / "scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py"
    trainer = code_root / "scripts/train_robotwin2_aloha_source_liquid_shared_event_head_v1.py"
    paired_runner = code_root / "scripts/run_robotwin2_aloha_smolvla_liquid_paired_v1.py"
    for path in (
        args.runtime_python,
        args.training_python,
        collector,
        trainer,
        paired_runner,
        args.actor_checkpoint,
        args.vlm_metadata_path,
        args.robotwin_root,
        args.event_spec,
        args.actor_execution_protocol,
        args.paired_seed_roster,
    ):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    if sha256_file(args.actor_execution_protocol.expanduser().resolve()) != (
        args.actor_execution_protocol_sha256
    ):
        raise PipelineError("actor execution protocol file changed")
    if sha256_file(args.paired_seed_roster.expanduser().resolve()) != (
        args.paired_seed_roster_sha256
    ):
        raise PipelineError("paired clean100 seed roster changed")
    protocol = json.loads(
        args.actor_execution_protocol.expanduser().resolve().read_text(
            encoding="utf-8"
        )
    )
    query_indices = [int(value) for value in protocol.get("query_indices", [])]
    target_per_stratum = int(protocol.get("target_per_condition_query", 0))
    expected_from_protocol = (
        2 * len(query_indices) * target_per_stratum
    )
    if (
        not query_indices
        or len(set(query_indices)) != len(query_indices)
        or target_per_stratum <= 0
        or expected_from_protocol != args.expected_groups
    ):
        raise PipelineError("pipeline group target disagrees with actor protocol")
    output_root.mkdir(parents=True, exist_ok=True)
    collector_base = [
        str(args.runtime_python.expanduser().resolve()),
        str(collector),
        "--body",
        SOURCE_BODY,
        "--actor-checkpoint",
        str(args.actor_checkpoint.expanduser().resolve()),
        "--vlm-metadata-path",
        str(args.vlm_metadata_path.expanduser().resolve()),
        "--robotwin-root",
        str(args.robotwin_root.expanduser().resolve()),
        "--event-spec",
        str(args.event_spec.expanduser().resolve()),
        "--actor-execution-protocol",
        str(args.actor_execution_protocol.expanduser().resolve()),
        "--actor-execution-protocol-sha256",
        args.actor_execution_protocol_sha256,
        "--path-root",
        str(args.path_root.expanduser().resolve()),
        "--output",
        str(source_root),
        "--actor-action-contract",
        args.actor_action_contract,
        "--liquid-history-length",
        str(args.history_length),
    ]
    update_state(
        state,
        "collecting_aloha_source",
        code_root=str(code_root),
        launcher_file_sha256=sha256_file(Path(__file__).resolve()),
        expected_decision_groups=args.expected_groups,
        expected_candidate_branches=args.expected_groups * 4,
        actor_action_contract=args.actor_action_contract,
        collector_seed_batch=args.collector_seed_batch,
        collection_manifest=str(manifest),
        training_output=str(training_root),
        paired_output=str(paired_root),
    )
    try:
        environment = collector_environment(
            code_root=code_root,
            path_root=path_root,
            robotwin_root=robotwin_root,
        )

        def invoke_collector(
            *,
            conditions: list[str],
            seed_start: int,
            seed_count: int | None = None,
            root_queries: list[int] | None = None,
        ) -> None:
            command = [
                *collector_base,
                "--conditions",
                *conditions,
                "--seed-start",
                str(seed_start),
            ]
            if seed_count is not None:
                command.extend(("--seed-count", str(seed_count)))
            if root_queries is not None:
                command.extend(
                    (
                        "--root-query-indices",
                        *(str(value) for value in root_queries),
                        "--manifest-root-query-indices",
                        *(str(value) for value in query_indices),
                    )
                )
            # RoboTwin 2.0 still resolves ``./assets`` relative to the process
            # working directory during import, independently of ASSETS_PATH.
            subprocess.run(
                command,
                cwd=robotwin_root,
                env=environment,
                check=True,
            )

        def quota_inventory(
            value: Mapping[str, Any],
        ) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int]]:
            counts = {
                (condition, query): 0
                for condition in ("clean", "randomized")
                for query in query_indices
            }
            maximum_seed = {
                key: args.seed_start - 1 for key in counts
            }
            groups = value.get("groups")
            if not isinstance(groups, list):
                raise PipelineError("collector manifest groups are invalid")
            for group in groups:
                if not isinstance(group, Mapping):
                    raise PipelineError("collector manifest group is invalid")
                key = (str(group.get("condition")), int(group.get("root_query_index", -1)))
                seed = group.get("requested_seed")
                if (
                    key not in counts
                    or isinstance(seed, bool)
                    or not isinstance(seed, int)
                ):
                    raise PipelineError("collector manifest stratum is invalid")
                counts[key] += 1
                maximum_seed[key] = max(maximum_seed[key], int(seed))
            if any(value > target_per_stratum for value in counts.values()):
                raise PipelineError("collector exceeded a frozen source quota")
            return counts, maximum_seed

        if manifest.is_file():
            manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
            counts, maximum_seed = quota_inventory(manifest_value)
        else:
            counts = {
                (condition, query): 0
                for condition in ("clean", "randomized")
                for query in query_indices
            }
            maximum_seed = {key: args.seed_start - 1 for key in counts}
        next_seed = {
            key: max(args.seed_start, maximum_seed[key] + 1)
            for key in counts
        }
        repair_invocations = 0
        consecutive_no_progress_failures = 0
        while any(value < target_per_stratum for value in counts.values()):
            repair_invocations += 1
            if repair_invocations > args.max_collector_invocations:
                raise PipelineError("source collection exceeded its invocation limit")
            key = next(
                key
                for key in sorted(counts)
                if counts[key] < target_per_stratum
            )
            condition, query = key
            deficit = target_per_stratum - counts[key]
            batch = min(deficit, args.collector_seed_batch)
            reserved_start = next_seed[key]
            next_seed[key] += batch
            before_total = sum(counts.values())
            update_state(
                state,
                "filling_aloha_source_stratum_quota",
                code_root=str(code_root),
                launcher_file_sha256=sha256_file(Path(__file__).resolve()),
                expected_decision_groups=args.expected_groups,
                collected_decision_groups=sum(counts.values()),
                active_condition=condition,
                active_root_query=query,
                active_stratum_deficit=deficit,
                active_reserved_seed_start=reserved_start,
                active_reserved_seed_count=batch,
                repair_invocations=repair_invocations,
                actor_action_contract=args.actor_action_contract,
                collector_seed_batch=args.collector_seed_batch,
                collection_manifest=str(manifest),
                training_output=str(training_root),
            )
            collector_error: subprocess.CalledProcessError | None = None
            try:
                invoke_collector(
                    conditions=[condition],
                    seed_start=reserved_start,
                    seed_count=batch,
                    root_queries=[query],
                )
            except subprocess.CalledProcessError as error:
                # Every group is atomically committed before the long-lived
                # simulator can abort.  A fresh subprocess may safely resume,
                # but repeated zero-progress failures indicate a real bug.
                collector_error = error
            if manifest.is_file():
                manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
                counts, maximum_seed = quota_inventory(manifest_value)
            after_total = sum(counts.values())
            if collector_error is not None and after_total == before_total:
                consecutive_no_progress_failures += 1
            else:
                consecutive_no_progress_failures = 0
            if (
                collector_error is not None
                and consecutive_no_progress_failures
                >= args.max_consecutive_collector_failures
            ):
                raise PipelineError(
                    "collector repeatedly exited without an atomic group commit: "
                    f"returncode={collector_error.returncode}"
                ) from collector_error
        if sum(counts.values()) != args.expected_groups or any(
            value != target_per_stratum for value in counts.values()
        ):
            raise PipelineError("collector did not close every Aloha source quota")
        manifest_sha = sha256_file(manifest)
        update_state(
            state,
            "training_liquid_v14_source_only",
            code_root=str(code_root),
            launcher_file_sha256=sha256_file(Path(__file__).resolve()),
            collected_decision_groups=args.expected_groups,
            collected_candidate_branches=args.expected_groups * 4,
            source_stratum_counts={
                f"{condition}|query={query}": count
                for (condition, query), count in sorted(counts.items())
            },
            source_quota_repair_invocations=repair_invocations,
            actor_action_contract=args.actor_action_contract,
            collector_seed_batch=args.collector_seed_batch,
            collection_manifest=str(manifest),
            collection_manifest_sha256=manifest_sha,
            training_output=str(training_root),
            paired_output=str(paired_root),
        )
        trainer_command = [
            str(args.training_python.expanduser().resolve()),
            str(trainer),
            "--manifest",
            str(manifest),
            "--manifest-sha256",
            manifest_sha,
            "--output",
            str(training_root),
            "--history-length",
            str(args.history_length),
            "--expected-groups",
            str(args.expected_groups),
            "--steps",
            str(args.steps),
            "--eval-every",
            str(args.eval_every),
            "--batch-size",
            str(args.batch_size),
        ]
        summary = training_root / "training_summary.json"
        if summary.is_file():
            existing_summary = json.loads(summary.read_text(encoding="utf-8"))
            if (
                existing_summary.get("status")
                != "source_only_training_complete_targets_still_sealed"
                or existing_summary.get("source_manifest_sha256") != manifest_sha
                or existing_summary.get("source_groups") != args.expected_groups
            ):
                raise PipelineError("existing liquid training summary changed")
        else:
            if training_root.exists():
                raise PipelineError(
                    "incomplete liquid training output exists and cannot be resumed"
                )
            subprocess.run(trainer_command, cwd=code_root, check=True)
        if not summary.is_file():
            raise PipelineError("trainer completed without a summary")
        summary_sha = sha256_file(summary)
        update_state(
            state,
            "paired_clean100_official_smolvla_vs_liquid",
            code_root=str(code_root),
            launcher_file_sha256=sha256_file(Path(__file__).resolve()),
            collected_decision_groups=args.expected_groups,
            collected_candidate_branches=args.expected_groups * 4,
            collection_manifest=str(manifest),
            collection_manifest_sha256=manifest_sha,
            training_output=str(training_root),
            training_summary=str(summary),
            training_summary_sha256=summary_sha,
            actor_action_contract=args.actor_action_contract,
            paired_output=str(paired_root),
            paired_seed_roster=str(args.paired_seed_roster.expanduser().resolve()),
            paired_seed_roster_sha256=args.paired_seed_roster_sha256,
        )
        paired_base = [
            str(args.runtime_python.expanduser().resolve()),
            str(paired_runner),
            "--actor-checkpoint",
            str(args.actor_checkpoint.expanduser().resolve()),
            "--vlm-metadata-path",
            str(args.vlm_metadata_path.expanduser().resolve()),
            "--robotwin-root",
            str(robotwin_root),
            "--event-spec",
            str(args.event_spec.expanduser().resolve()),
            "--actor-execution-protocol",
            str(args.actor_execution_protocol.expanduser().resolve()),
            "--actor-execution-protocol-sha256",
            args.actor_execution_protocol_sha256,
            "--training-summary",
            str(summary),
            "--training-summary-sha256",
            summary_sha,
            "--seed-roster",
            str(args.paired_seed_roster.expanduser().resolve()),
            "--seed-roster-sha256",
            args.paired_seed_roster_sha256,
            "--output",
            str(paired_root),
        ]
        for start in range(0, 100, args.paired_seed_batch):
            pair_paths = [
                paired_root / "pairs" / f"seed_{seed}.json"
                for seed in json.loads(
                    args.paired_seed_roster.expanduser().resolve().read_text(
                        encoding="utf-8"
                    )
                )["seeds"][start : start + args.paired_seed_batch]
            ]
            if all(path.is_file() for path in pair_paths):
                continue
            failures = 0
            while True:
                try:
                    subprocess.run(
                        [
                            *paired_base,
                            "--seed-index-start",
                            str(start),
                            "--seed-index-count",
                            str(args.paired_seed_batch),
                        ],
                        cwd=robotwin_root,
                        env=environment,
                        check=True,
                    )
                    break
                except subprocess.CalledProcessError:
                    failures += 1
                    if failures >= args.max_consecutive_collector_failures:
                        raise
        subprocess.run(
            [*paired_base, "--finalize-only"],
            cwd=robotwin_root,
            env=environment,
            check=True,
        )
        paired_report = paired_root / "paired_report.json"
        if not paired_report.is_file():
            raise PipelineError("paired comparison completed without a report")
        update_state(
            state,
            "training_and_paired_clean100_complete",
            code_root=str(code_root),
            launcher_file_sha256=sha256_file(Path(__file__).resolve()),
            collected_decision_groups=args.expected_groups,
            collected_candidate_branches=args.expected_groups * 4,
            collection_manifest=str(manifest),
            collection_manifest_sha256=manifest_sha,
            training_output=str(training_root),
            training_summary=str(summary),
            training_summary_sha256=summary_sha,
            actor_action_contract=args.actor_action_contract,
            paired_output=str(paired_root),
            paired_report=str(paired_report),
            paired_report_sha256=sha256_file(paired_report),
            next_required_stage="cross_embodiment_heldout_body_evaluation",
        )
    except BaseException as error:
        update_state(
            state,
            "failed",
            code_root=str(code_root),
            launcher_file_sha256=sha256_file(Path(__file__).resolve()),
            error_type=type(error).__name__,
            error=str(error),
            actor_action_contract=args.actor_action_contract,
            collection_manifest=str(manifest),
            training_output=str(training_root),
            paired_output=str(paired_root),
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--training-python", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--actor-action-contract",
        choices=("ee16", "aloha_joint14"),
        default="aloha_joint14",
    )
    parser.add_argument("--vlm-metadata-path", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol", type=Path, required=True)
    parser.add_argument("--actor-execution-protocol-sha256", required=True)
    parser.add_argument("--paired-seed-roster", type=Path, required=True)
    parser.add_argument("--paired-seed-roster-sha256", required=True)
    parser.add_argument("--seed-start", type=int, default=2026082000)
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--expected-groups", type=int, default=400)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--collector-seed-batch", type=int, default=4)
    parser.add_argument("--paired-seed-batch", type=int, default=4)
    parser.add_argument("--max-collector-invocations", type=int, default=2000)
    parser.add_argument(
        "--max-consecutive-collector-failures", type=int, default=3
    )
    args = parser.parse_args()
    if (
        args.history_length < 2
        or args.expected_groups <= 0
        or args.steps <= 0
        or args.eval_every <= 0
        or args.batch_size < 4
        or args.collector_seed_batch <= 0
        or args.paired_seed_batch <= 0
        or args.paired_seed_batch > 100
        or args.max_collector_invocations <= 0
        or args.max_consecutive_collector_failures <= 0
    ):
        parser.error("invalid source-only pipeline dimensions/budget")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
