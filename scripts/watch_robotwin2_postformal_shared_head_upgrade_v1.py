#!/usr/bin/env python3
"""Run the complete C+expert-root shared-head upgrade after the formal pipeline.

This CPU-only detached watcher deliberately waits for the existing formal
collection, C-only LOBO, paired-success study, and postformal ablation to finish
before reserving the single RTX 4090.  It then runs, in order:

1. the complete five-body 150-decision/600-branch scripted e12/e3/e4 supplement;
2. its immutable five-body binding;
3. five strict C+supplement outer-LOBO folds;
4. one full 1,000-initial-condition actor/N4/N8 shared-raw16 nested study.

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
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_robotwin2_postformal_shared_head_upgrade_watcher_v2"
RECOVERABLE_INTERRUPTION_STATUS = "recoverable_child_signal_interruption"
RECOVERABLE_WATCHER_EXIT_CODE = 75
RECOVERABLE_INTERRUPTION_SIGNALS = frozenset(
    int(member)
    for member in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGKILL)
)
UPSTREAM_FORMAT = "etsf_robotwin2_five_body_postformal_ablation_watcher_v1"
SUPPLEMENT_MANIFEST_FORMAT = (
    "etsf_robotwin2_proper_world_utility_rank_supplement_manifest_v2"
)
SUPPLEMENT_BINDING_FORMAT = (
    "etsf_robotwin2_five_body_proper_world_utility_rank_supplement_binding_v2"
)
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
CONDITIONS = ("clean", "randomized")
TARGET_EVENTS = ("e12", "e3", "e4")
SUPPLEMENT_HORIZONS = (10, 25, 50, 100, 200)
SUPPLEMENT_RESERVE_SEED_START = 2026081000
SUPPLEMENT_RESERVE_SEEDS_PER_SLOT = 16
SUPPLEMENT_RESERVE_SEED_STOP_EXCLUSIVE = 2026081800
EXPECTED_SUPPLEMENT_DECISIONS_PER_BODY = 30
EXPECTED_SUPPLEMENT_DECISIONS = 150
EXPECTED_SUPPLEMENT_BRANCHES = 600
N8_RETAINED_CANDIDATE_COUNT = 8
N8_RAW_PROPOSAL_COUNT = 16
NESTED_RUNNER_FORMAT = "etsf_robotwin2_five_body_nested_n4_n8_execution_v2"
NESTED_CONTRACT_FORMAT = "etsf_robotwin2_nested_n4_n8_execution_contract_v1"
NESTED_OUTCOME_FORMAT = "etsf_robotwin2_nested_n4_n8_outcomes_v2"
NESTED_REPORT_FORMAT = "etsf_robotwin2_nested_n4_n8_report_v2"
NESTED_COMPLETION_FORMAT = (
    "etsf_robotwin2_nested_n4_n8_completion_receipt_v2"
)
NESTED_PROTOCOL_FORMAT = "etsf_robotwin2_nested_n4_n8_prospective_protocol_v1"
NESTED_METHODS = (
    "actor_baseline",
    "etsf_nested_best_of_4_from_raw16",
    "etsf_nested_best_of_8_from_raw16",
)
NESTED_SEED_BASE = 2026091000
NESTED_SEED_COUNT = 100
EXPECTED_NESTED_TRIPLETS = 1000
EXPECTED_NESTED_ROLLOUTS = 3000
EXPECTED_GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"
DEFAULT_ETSF_SITE = Path(
    "/home/user/anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages"
)


class SharedHeadUpgradeError(RuntimeError):
    """The upstream, supplement, training, or paired upgrade failed closed."""


class RecoverableChildSignalInterruption(SharedHeadUpgradeError):
    """A stage child was unambiguously stopped by an interruption signal."""

    def __init__(self, stage: str, returncode: int) -> None:
        if (
            isinstance(returncode, bool)
            or returncode >= 0
            or -returncode not in RECOVERABLE_INTERRUPTION_SIGNALS
        ):
            raise ValueError("recoverable interruption requires an allowed signal")
        self.stage = stage
        self.child_returncode = returncode
        self.signal_number = -returncode
        self.signal_name = signal.Signals(self.signal_number).name
        super().__init__(
            f"{stage} interrupted by {self.signal_name} ({self.signal_number})"
        )


def raise_for_stage_returncode(stage: str, returncode: int) -> None:
    """Classify only direct POSIX interruption signals as recoverable."""

    if returncode == 0:
        return
    if (
        not isinstance(returncode, bool)
        and returncode < 0
        and -returncode in RECOVERABLE_INTERRUPTION_SIGNALS
    ):
        raise RecoverableChildSignalInterruption(stage, returncode)
    raise SharedHeadUpgradeError(f"{stage} exited {returncode}")


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


def verify_named_sha(
    value: Mapping[str, Any], field: str, label: str
) -> None:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if declared != canonical_sha256(unsigned):
        raise SharedHeadUpgradeError(f"{label} {field} mismatch")


def validate_nested_protocol(value: Mapping[str, Any]) -> str:
    verify_logical_sha(value, "nested evaluation protocol")
    if (
        value.get("format") != NESTED_PROTOCOL_FORMAT
        or value.get("evaluation_seed_base") != NESTED_SEED_BASE
        or value.get("evaluation_seed_count") != NESTED_SEED_COUNT
        or value.get("formal_seed_block_reused") is not False
        or value.get("seed_block_selected_before_any_nested_rollout_outcome")
        is not True
        or value.get("balanced_body_condition_cells")
        != len(BODIES) * len(CONDITIONS)
        or value.get("bootstrap_unit")
        != "requested_seed_cluster_with_all_selected_body_condition_rows_kept_together"
        or value.get("pooled_mcnemar_role")
        != "descriptive_only_due_repeated_requested_seeds"
        or value.get("single_body_condition_mcnemar_role") != "inferential"
        or not isinstance(value.get("bootstrap_seed_derivation"), Mapping)
    ):
        raise SharedHeadUpgradeError("nested evaluation protocol changed")
    return str(value["logical_sha256"])


def validate_nested_completion(root: Path) -> dict[str, Any]:
    """Validate the complete contract→outcomes→report→receipt SHA chain."""

    root = root.expanduser().resolve()
    contract_path = root / "execution_contract.json"
    outcome_path = root / "nested_paired_outcomes.json"
    report_path = root / "nested_n4_n8_report.json"
    completion_path = root / "completion_receipt.json"
    contract = read_json(contract_path, "nested execution contract")
    outcome = read_json(outcome_path, "nested outcomes")
    report = read_json(report_path, "nested report")
    completion = read_json(completion_path, "nested completion receipt")

    verify_logical_sha(contract, "nested execution contract")
    protocol = contract.get("nested_evaluation_protocol")
    if not isinstance(protocol, Mapping):
        raise SharedHeadUpgradeError("nested contract lacks its evaluation protocol")
    protocol_sha = validate_nested_protocol(protocol)
    persistence = contract.get("method_result_persistence")
    if (
        contract.get("format") != NESTED_CONTRACT_FORMAT
        or contract.get("runner_format") != NESTED_RUNNER_FORMAT
        or contract.get("bodies") != list(BODIES)
        or contract.get("conditions") != list(CONDITIONS)
        or contract.get("evaluation_seed_base") != NESTED_SEED_BASE
        or contract.get("evaluation_seed_count") != NESTED_SEED_COUNT
        or contract.get("initial_condition_triplet_count")
        != EXPECTED_NESTED_TRIPLETS
        or contract.get("rollout_count") != EXPECTED_NESTED_ROLLOUTS
        or contract.get("methods") != list(NESTED_METHODS)
        or contract.get("same_requested_seed_and_complete_reset_tripled") is not True
        or contract.get("method_order_rotated_before_outcomes") is not True
        or contract.get("no_training") is not True
        or not isinstance(persistence, Mapping)
        or persistence.get("existing_result_overwrite_or_retry_allowed") is not False
        or persistence.get("automatic_noninformative_resume_limit_per_method")
        != 1
        or persistence.get("exception_or_action_failure_retry_allowed") is not False
        or persistence.get("later_method_before_complete_prefix_allowed") is not False
    ):
        raise SharedHeadUpgradeError("nested execution contract is incomplete")
    contract_file_sha = sha256_file(contract_path)

    verify_named_sha(outcome, "document_sha256", "nested outcomes")
    rows = outcome.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_NESTED_TRIPLETS:
        raise SharedHeadUpgradeError("nested outcomes lack exactly 1000 rows")
    expected_identities = {
        (body, condition, NESTED_SEED_BASE + ordinal)
        for body in BODIES
        for condition in CONDITIONS
        for ordinal in range(NESTED_SEED_COUNT)
    }
    observed_identities: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SharedHeadUpgradeError(f"nested outcome row {index} is invalid")
        identity = (
            row.get("heldout_body"),
            row.get("condition"),
            row.get("requested_seed"),
        )
        if identity not in expected_identities or identity in observed_identities:
            raise SharedHeadUpgradeError("nested outcome identity roster changed")
        observed_identities.add(identity)
        order = row.get("method_order")
        if (
            not isinstance(order, list)
            or len(order) != len(NESTED_METHODS)
            or set(order) != set(NESTED_METHODS)
        ):
            raise SharedHeadUpgradeError("nested outcome method order changed")
        for method in NESTED_METHODS:
            success = row.get(f"{method}_binary_success")
            stage = row.get(f"{method}_stage_progress")
            if (
                type(success) is not int
                or success not in (0, 1)
                or isinstance(stage, bool)
                or not isinstance(stage, (int, float))
                or not 0.0 <= float(stage) <= 1.0
            ):
                raise SharedHeadUpgradeError("nested outcome value changed")
    outcome_protocol = outcome.get("nested_evaluation_protocol")
    if (
        observed_identities != expected_identities
        or outcome.get("format") != NESTED_OUTCOME_FORMAT
        or outcome.get("status")
        != "complete_1000_initial_condition_triplets_3000_rollouts"
        or outcome.get("pair_count") != EXPECTED_NESTED_TRIPLETS
        or outcome.get("rollout_count") != EXPECTED_NESTED_ROLLOUTS
        or outcome.get("methods") != list(NESTED_METHODS)
        or outcome.get("rows_sha256") != canonical_sha256(rows)
        or outcome.get("execution_contract_logical_sha256")
        != contract.get("logical_sha256")
        or outcome.get("execution_contract_file_sha256") != contract_file_sha
        or not isinstance(outcome_protocol, Mapping)
        or dict(outcome_protocol) != dict(protocol)
    ):
        raise SharedHeadUpgradeError("nested outcome document binding changed")
    outcome_file_sha = sha256_file(outcome_path)

    verify_named_sha(report, "report_sha256", "nested report")
    report_protocol = report.get("nested_evaluation_protocol")
    expected_cells = {
        f"{body}|{condition}" for body in BODIES for condition in CONDITIONS
    }
    if (
        report.get("format") != NESTED_REPORT_FORMAT
        or report.get("status")
        != "complete_shared_raw16_nested_n4_n8_paired_report"
        or report.get("outcome_document_sha256")
        != outcome.get("document_sha256")
        or not isinstance(report_protocol, Mapping)
        or dict(report_protocol) != dict(protocol)
        or set(report.get("by_heldout_body", {})) != set(BODIES)
        or set(report.get("by_heldout_body_and_condition", {})) != expected_cells
    ):
        raise SharedHeadUpgradeError("nested report binding changed")
    report_file_sha = sha256_file(report_path)

    verify_logical_sha(completion, "nested completion receipt")
    if (
        completion.get("format") != NESTED_COMPLETION_FORMAT
        or completion.get("status")
        != "complete_1000_triplets_3000_rollouts_frozen"
        or completion.get("execution_contract_logical_sha256")
        != contract.get("logical_sha256")
        or completion.get("execution_contract_file_sha256") != contract_file_sha
        or completion.get("outcome_document_sha256")
        != outcome.get("document_sha256")
        or completion.get("outcome_file_sha256") != outcome_file_sha
        or completion.get("report_sha256") != report.get("report_sha256")
        or completion.get("report_file_sha256") != report_file_sha
        or completion.get("initial_condition_triplet_count")
        != EXPECTED_NESTED_TRIPLETS
        or completion.get("rollout_count") != EXPECTED_NESTED_ROLLOUTS
        or completion.get("nested_evaluation_protocol_logical_sha256")
        != protocol_sha
    ):
        raise SharedHeadUpgradeError("nested completion SHA chain changed")
    return {
        "contract_file_sha256": contract_file_sha,
        "outcome_file_sha256": outcome_file_sha,
        "report_file_sha256": report_file_sha,
        "completion_file_sha256": sha256_file(completion_path),
        "nested_evaluation_protocol_logical_sha256": protocol_sha,
        "completed_initial_condition_triplets": EXPECTED_NESTED_TRIPLETS,
        "completed_rollouts": EXPECTED_NESTED_ROLLOUTS,
        "completed_rollouts_by_method": {
            method: EXPECTED_NESTED_TRIPLETS for method in NESTED_METHODS
        },
        "report": str(report_path),
    }


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
        str(N8_RETAINED_CANDIDATE_COUNT),
        "--proposal-count",
        str(N8_RAW_PROPOSAL_COUNT),
        "--output",
        str(args.augmented_n8_root),
        "--action-exec-steps",
        "5",
        "--max-steps",
        "200",
        "--fps",
        "15.0",
    ]


def nested_n4_n8_command(
    args: argparse.Namespace, supplement_sha256: str
) -> list[str]:
    """Run the strong shared-raw16 actor/N4/N8 comparison in one job."""

    return [
        str(args.robotwin_python),
        str(
            args.code_root
            / "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py"
        ),
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
        raise_for_stage_returncode(stage, result.returncode)

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
        "run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py",
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
            raise SharedHeadUpgradeError(f"{body} supplement did not complete 30 decisions")
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
        raise SharedHeadUpgradeError("supplement binding is not the complete 150/600 design")
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

    wait_idle("waiting_for_idle_rtx4090_before_nested_actor_n4_n8")
    run_stage(
        "running_complete_shared_raw16_nested_actor_n4_n8_evaluation",
        nested_n4_n8_command(args, supplement_sha256),
        "nested_actor_n4_n8_paired.log",
        cwd=args.robotwin_root,
        environment=environment,
    )
    nested_completion = validate_nested_completion(args.augmented_n8_root)
    nested_report = Path(str(nested_completion["report"]))

    atomic_text(args.run_exit, "0\n")
    write_state(
        "complete",
        supplement_binding_sha256=supplement_sha256,
        augmented_lobo_summary=str(
            args.augmented_lobo_root / "five_fold_training_summary.json"
        ),
        nested_actor_n4_n8_report=str(nested_report),
        nested_actor_n4_n8_report_file_sha256=nested_completion[
            "report_file_sha256"
        ],
        nested_completion_audit=nested_completion,
        completed_initial_condition_triplets=nested_completion[
            "completed_initial_condition_triplets"
        ],
        completed_rollouts_by_method=nested_completion[
            "completed_rollouts_by_method"
        ],
        n4_is_exact_ordered_prefix_of_n8=True,
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


def record_recoverable_interruption(
    state: Path | None, error: RecoverableChildSignalInterruption
) -> None:
    """Persist a narrow guardian restart authorization without terminal failure."""

    if state is not None:
        atomic_json(
            state,
            {
                "format": FORMAT,
                "status": RECOVERABLE_INTERRUPTION_STATUS,
                "updated_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "child_stage": error.stage,
                "child_returncode": error.child_returncode,
                "child_signal_number": error.signal_number,
                "child_signal_name": error.signal_name,
                "run_exit_written": False,
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
    except RecoverableChildSignalInterruption as error:
        record_recoverable_interruption(state_path, error)
        raise SystemExit(RECOVERABLE_WATCHER_EXIT_CODE)
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            record_failure(state_path, run_exit_path, error)
        raise


__all__ = [
    "RecoverableChildSignalInterruption",
    "RECOVERABLE_INTERRUPTION_SIGNALS",
    "RECOVERABLE_INTERRUPTION_STATUS",
    "RECOVERABLE_WATCHER_EXIT_CODE",
    "SharedHeadUpgradeError",
    "evaluator_command",
    "fold_arguments",
    "lobo_command",
    "N8_RAW_PROPOSAL_COUNT",
    "N8_RETAINED_CANDIDATE_COUNT",
    "materializer_command",
    "nested_n4_n8_command",
    "paired_n4_command",
    "paired_n8_command",
    "raise_for_stage_returncode",
    "record_recoverable_interruption",
    "supplement_collector_command",
    "supplement_manifest_complete",
    "supplement_reserve_roster",
    "validate_nested_completion",
    "validate_upstream_state",
]
