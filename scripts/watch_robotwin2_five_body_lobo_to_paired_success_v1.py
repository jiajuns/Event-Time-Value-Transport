#!/usr/bin/env python3
"""Wait for formal five-fold LOBO completion, then run paired success v1.

This is a remote, detached orchestration entry point.  It uses only the Python
standard library while waiting, validates the upstream terminal state and all
25 selected LOBO checkpoints, waits without reserving CUDA until the authorized
RTX 4090 is idle, runs the fixed 1,000-pair/2,000-rollout command, and finally
invokes the frozen standard-library evaluator.  Paths that define the formal
experiment are constants rather than command-line options.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


FORMAT = "etsf_robotwin2_five_body_lobo_to_paired_success_watcher_v1"
BINDING_FORMAT = "etsf_robotwin2_five_body_paired_success_execution_binding_v1"
UPSTREAM_FORMAT = "etsf_robotwin2_five_body_branches_to_lobo_watcher_v1"
UPSTREAM_FINAL_FORMAT = "etsf_robotwin2_five_body_lobo_source_validation_aggregate_v1"
UPSTREAM_FINAL_STATUS = "five_outer_lobo_source_only_training_complete"
FOLD_FORMAT = "etsf_robotwin2_five_body_lobo_shared_event_head_v1"
FOLD_STATUS = "source_only_checkpoint_selection_complete"
REPORT_FORMAT = "etsf_robotwin2_move_can_pot_cross_embodiment_paired_success_report_v2"
REPORT_STATUS = "metrics_computed_no_promotion_deployment_or_claim_authority"
EXPECTED_GPU_INDEX = 0
EXPECTED_GPU_UUID = "GPU-06f6e50e-5296-258f-dd86-8f838390a7d1"
# Frozen code/data identities for the formal remote continuation chain.
EXPECTED_EVENT_SPEC_SHA256 = (
    "4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee"
)
EXPECTED_EVENT_MODULE_SHA256 = (
    "d236036e4121232391808743a957e8ae94722ea89df223d123f8a77296f9e6d9"
)
EXPECTED_RUNNER_SHA256 = (
    "98463a7979d1fb88cefddecf07548069ec51b6731b15d1777eb4f90ef7e50648"
)
EXPECTED_EVALUATOR_SHA256 = (
    "6e0f2a9b370f6c8fb66caf8c01e55747f4b882ced3657a1a2b32346d9bda9984"
)
EXPECTED_MATERIALIZATION_RECEIPT_FILE_SHA256 = (
    "aefd33cd337dbaad5d85e6a7cf5490221cb515fe6bb06462257d279a091f8582"
)
EXPECTED_MATERIALIZATION_PREREGISTRATION_SHA256 = (
    "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee"
)
EXPECTED_METRICS_PREREGISTRATION_SHA256 = (
    "a4e59f647c520609313e1c9aca03dbb3f770504e0383c66bb619dca94b4c6827"
)
MATERIALIZATION_FORMAT = "etsf_robotwin2_move_can_pot_public_materialization_receipt_v1"
MATERIALIZATION_STATUS = (
    "verified_complete_public_payload_materialization_no_operational_authority"
)
METRICS_PREREGISTRATION_FORMAT = (
    "etsf_robotwin2_move_can_pot_five_body_lobo_preregistration_v2"
)
METRICS_PREREGISTRATION_STATUS = (
    "preregistered_data_blind_no_download_training_evaluation_or_claim"
)
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
EXPECTED_PAIRS = 1_000
EXPECTED_ROLLOUTS = 2_000
EXPECTED_MEMBERS_PER_FOLD = 5
ACTION_EXEC_STEPS = 5
MAX_STEPS = 200
FPS = 15
STANDARDIZED_RANK_ENSEMBLE_CONTRACT = {
    "format": "etsf_within_decision_standardized_rank_ensemble_v1",
    "member_count": 5,
    "candidate_count": 4,
    "member_transform": "subtract_candidate_mean_divide_population_std",
    "population_std_correction": 0,
    "std_floor": 1e-6,
    "member_with_std_at_or_below_floor": "all_zero_contribution",
    "aggregation": "equal_mean_over_exactly_five_member_contributions",
    "normalization_scope": "one_four_candidate_decision_per_member",
}


class PairedWatcherError(RuntimeError):
    """The upstream, static input, GPU, execution, or report contract failed."""


@dataclass(frozen=True)
class FormalPaths:
    home: Path
    code_root: Path
    upstream_state: Path
    upstream_run_exit: Path
    lobo_root: Path
    actor_checkpoint: Path
    vlm_metadata: Path
    robotwin_root: Path
    event_spec: Path
    materialization_receipt: Path
    metrics_preregistration: Path
    output_root: Path
    state: Path
    run_exit: Path
    pid: Path
    watcher_log: Path
    instance_lock: Path
    gpu_lock: Path
    runner_python: Path
    evaluator_python: Path
    lerobot_root: Path
    lerobot_site: Path
    robotwin_eval_site: Path

    @property
    def runner(self) -> Path:
        return self.code_root / "run_robotwin2_five_body_paired_success_v1.py"

    @property
    def evaluator(self) -> Path:
        return self.code_root / "evaluate_robotwin2_cross_embodiment_paired_success_v1.py"

    @property
    def event_module(self) -> Path:
        return self.code_root / "robotwin2_move_can_pot_analytic_event_spec_v1.py"

    @property
    def final_summary(self) -> Path:
        return self.lobo_root / "five_fold_training_summary.json"

    @property
    def execution_binding(self) -> Path:
        return self.output_root / "watcher_execution_binding.json"

    @property
    def execution_contract(self) -> Path:
        return self.output_root / "execution_contract.json"

    @property
    def outcomes(self) -> Path:
        return self.output_root / "paired_outcomes.json"

    @property
    def completion_receipt(self) -> Path:
        return self.output_root / "paired_execution_completion_receipt.json"

    @property
    def report(self) -> Path:
        return self.output_root / "paired_success_report.json"

    @property
    def runner_log(self) -> Path:
        return self.output_root / "logs" / "paired_execution.log"

    @property
    def evaluator_log(self) -> Path:
        return self.output_root / "logs" / "paired_evaluator.log"

    def fold_root(self, body: str) -> Path:
        return self.lobo_root / f"outer_lobo_{body}"


def formal_paths(code_root: Path | None = None) -> FormalPaths:
    home = Path("/home/user")
    paired_prefix = home / (
        "etsf_robotwin2_fivebody_paired_success_full2000_20260830_v2_analytic"
    )
    resolved_code = (code_root or Path(__file__).resolve().parent).resolve()
    return FormalPaths(
        home=home,
        code_root=resolved_code,
        upstream_state=home
        / (
            "etsf_robotwin2_fivebody_lobo_full8000_20260830_v2_analytic."
            "watcher_state.json"
        ),
        upstream_run_exit=home
        / "etsf_robotwin2_fivebody_lobo_full8000_20260830_v2_analytic.run.exit",
        lobo_root=home
        / "etsf_robotwin2_fivebody_lobo_shared_head_full8000_20260830_v2_analytic",
        actor_checkpoint=home
        / (
            "etsf_smolvla_models/"
            "smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/"
            "checkpoints/020000/pretrained_model"
        ),
        vlm_metadata=home / "etsf_stage0/offline_assets/smolvlm2_500m_metadata",
        robotwin_root=home / "etsf_stage0/RoboTwin",
        event_spec=home
        / "etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json",
        materialization_receipt=home
        / (
            "public_benchmark_receipts/"
            "robotwin2_move_can_pot_5emb_materialization_a967b852_20260830_v1.json"
        ),
        metrics_preregistration=home
        / (
            "public_benchmark_receipts/"
            "robotwin2_move_can_pot_5emb_metrics_preregistration_20260830_v2.json"
        ),
        output_root=paired_prefix,
        state=Path(str(paired_prefix) + ".watcher_state.json"),
        run_exit=Path(str(paired_prefix) + ".run.exit"),
        pid=Path(str(paired_prefix) + ".watcher.pid"),
        watcher_log=Path(str(paired_prefix) + ".watcher.log"),
        instance_lock=Path(str(paired_prefix) + ".watcher.lock"),
        gpu_lock=Path(str(paired_prefix) + ".gpu.lock"),
        runner_python=home / "anaconda3/envs/RoboTwin2/bin/python",
        evaluator_python=Path("/usr/bin/python3"),
        lerobot_root=home / "etsf_stage0/lerobot",
        lerobot_site=home
        / "etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages",
        robotwin_eval_site=home
        / "etsf_stage0/.venv_smolvla_robotwin_eval_np126/lib/python3.10/site-packages",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path, label: str) -> dict[str, Any]:
    root = path.resolve(strict=True)
    if path.is_symlink() or not root.is_dir():
        raise PairedWatcherError(f"{label} must be a real directory: {path}")
    rows: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise PairedWatcherError(f"{label} contains a symbolic link: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise PairedWatcherError(f"{label} contains a special file: {candidate}")
        rows.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
        )
    if not rows:
        raise PairedWatcherError(f"{label} is empty: {root}")
    return {
        "path": str(root),
        "tree_sha256": canonical_sha256(rows),
        "file_count": len(rows),
        "size_bytes": sum(row["size_bytes"] for row in rows),
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PairedWatcherError(f"{label} must be a real JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PairedWatcherError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise PairedWatcherError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_once_or_verify(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists() or path.is_symlink():
        if read_json(path, label) != value:
            raise PairedWatcherError(f"existing {label} differs from frozen value")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".create", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path, label) != value:
                raise PairedWatcherError(f"racing {label} differs from frozen value")
    finally:
        temporary.unlink(missing_ok=True)


def _contained_real_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise PairedWatcherError(f"{label} must be an absolute path")
    candidate = Path(raw)
    if candidate.is_symlink() or not candidate.is_file():
        raise PairedWatcherError(f"{label} is missing or symbolic: {candidate}")
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise PairedWatcherError(f"{label} escapes its fold root") from error
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PairedWatcherError(f"{label} contains a symbolic-link component")
    return resolved


def inspect_fold(paths: FormalPaths, body: str) -> dict[str, Any]:
    root = paths.fold_root(body)
    if root.is_symlink() or not root.is_dir():
        raise PairedWatcherError(f"LOBO fold is missing or symbolic for {body}: {root}")
    summary_path = root / "training_summary.json"
    summary = read_json(summary_path, f"{body} training summary")
    members = summary.get("members")
    if (
        summary.get("format") != FOLD_FORMAT
        or summary.get("status") != FOLD_STATUS
        or summary.get("held_out_body") != body
        or summary.get("event_spec_sha256") != EXPECTED_EVENT_SPEC_SHA256
        or summary.get("event_derivation_implementation_sha256")
        != EXPECTED_EVENT_MODULE_SHA256
        or summary.get("heldout_labels_used_for_normalization_training_or_selection")
        is not False
        or summary.get("heldout_specific_trainable_parameters") != 0
        or summary.get("actor_frozen") is not True
        or not isinstance(members, list)
        or len(members) != EXPECTED_MEMBERS_PER_FOLD
    ):
        raise PairedWatcherError(f"LOBO training summary contract changed for {body}")
    normalized: list[dict[str, Any]] = []
    identities: set[int] = set()
    for item in members:
        if not isinstance(item, Mapping):
            raise PairedWatcherError(f"{body} fold member is not an object")
        member = item.get("member")
        if type(member) is not int or member in identities:
            raise PairedWatcherError(f"{body} fold member identity is invalid")
        checkpoint = _contained_real_file(
            root, item.get("checkpoint"), f"{body} member {member} checkpoint"
        )
        observed = sha256_file(checkpoint)
        if observed != item.get("checkpoint_sha256"):
            raise PairedWatcherError(f"{body} member {member} checkpoint SHA mismatch")
        seed = item.get("seed")
        if type(seed) is not int:
            raise PairedWatcherError(f"{body} member {member} seed is invalid")
        identities.add(member)
        normalized.append(
            {
                "member": member,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed,
            }
        )
    if identities != set(range(EXPECTED_MEMBERS_PER_FOLD)):
        raise PairedWatcherError(f"{body} fold members must be exactly 0..4")
    return {
        "heldout_body": body,
        "fold_root": str(root.resolve()),
        "training_summary": str(summary_path.resolve()),
        "training_summary_sha256": sha256_file(summary_path),
        "event_spec_sha256": EXPECTED_EVENT_SPEC_SHA256,
        "event_derivation_implementation_sha256": EXPECTED_EVENT_MODULE_SHA256,
        "members": sorted(normalized, key=lambda row: row["member"]),
    }


def validate_upstream_complete(paths: FormalPaths, state: Mapping[str, Any]) -> dict[str, Any]:
    if (
        state.get("format") != UPSTREAM_FORMAT
        or state.get("status") != "complete"
        or state.get("output_root") != str(paths.lobo_root)
        or state.get("actor_checkpoint") != str(paths.actor_checkpoint)
        or state.get("completed_folds") != list(BODIES)
        or state.get("final_summary") != str(paths.final_summary)
    ):
        raise PairedWatcherError("upstream complete-state contract changed")
    if not paths.upstream_run_exit.is_file() or paths.upstream_run_exit.is_symlink():
        raise PairedWatcherError("upstream claims complete without a real run.exit")
    if paths.upstream_run_exit.read_text(encoding="utf-8") != "0\n":
        raise PairedWatcherError("upstream run.exit is not zero")
    final = read_json(paths.final_summary, "five-fold aggregate")
    final_sha = sha256_file(paths.final_summary)
    if (
        state.get("final_summary_file_sha256") != final_sha
        or final.get("format") != UPSTREAM_FINAL_FORMAT
        or final.get("status") != UPSTREAM_FINAL_STATUS
        or final.get("fold_count") != len(BODIES)
        or final.get("members_per_fold") != EXPECTED_MEMBERS_PER_FOLD
        or final.get("heldout_task_success_measured") is not False
        or final.get("cross_embodiment_task_success_claim_authorized") is not False
    ):
        raise PairedWatcherError("five-fold aggregate contract changed")
    folds = {body: inspect_fold(paths, body) for body in BODIES}
    aggregate_folds = final.get("outer_folds")
    if not isinstance(aggregate_folds, list) or len(aggregate_folds) != len(BODIES):
        raise PairedWatcherError("five-fold aggregate does not contain five outer folds")
    aggregate_by_body = {
        row.get("held_out_body"): row
        for row in aggregate_folds
        if isinstance(row, Mapping)
    }
    if set(aggregate_by_body) != set(BODIES):
        raise PairedWatcherError("five-fold aggregate held-out identities changed")
    for body, fold in folds.items():
        aggregate = aggregate_by_body[body]
        if (
            aggregate.get("training_summary") != fold["training_summary"]
            or aggregate.get("training_summary_file_sha256")
            != fold["training_summary_sha256"]
            or aggregate.get("member_count") != EXPECTED_MEMBERS_PER_FOLD
        ):
            raise PairedWatcherError(f"aggregate/fold binding changed for {body}")
    return {
        "upstream_state_file_sha256": sha256_file(paths.upstream_state),
        "upstream_run_exit_file_sha256": sha256_file(paths.upstream_run_exit),
        "final_summary": str(paths.final_summary),
        "final_summary_file_sha256": final_sha,
        "folds": folds,
    }


def probe_upstream(paths: FormalPaths) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not paths.upstream_state.exists():
        return None, {"upstream_state_present": False}
    state = read_json(paths.upstream_state, "upstream watcher state")
    status_value = state.get("status")
    if status_value == "failed":
        raise PairedWatcherError(
            f"upstream watcher failed: {state.get('error', 'unspecified error')}"
        )
    if paths.upstream_run_exit.exists():
        exit_value = paths.upstream_run_exit.read_text(encoding="utf-8")
        if exit_value != "0\n":
            raise PairedWatcherError(f"upstream watcher run.exit is {exit_value!r}")
    if status_value != "complete":
        return None, {
            "upstream_state_present": True,
            "upstream_status": status_value,
            "upstream_updated_at_utc": state.get("updated_at_utc"),
        }
    return validate_upstream_complete(paths, state), {
        "upstream_state_present": True,
        "upstream_status": "complete",
        "upstream_updated_at_utc": state.get("updated_at_utc"),
    }


def code_binding(paths: FormalPaths) -> list[dict[str, Any]]:
    names = (
        "watch_robotwin2_five_body_lobo_to_paired_success_v1.py",
        "run_robotwin2_five_body_paired_success_v1.py",
        "evaluate_robotwin2_cross_embodiment_paired_success_v1.py",
        "collect_robotwin2_five_body_ee_candidate_branches_v1.py",
        "preregister_robotwin2_move_can_pot_five_body_lobo_v1.py",
        "train_robotwin2_five_body_lobo_shared_event_head_v1.py",
        "robotwin2_cross_body_canonical_adapter_v1.py",
        "train_multibody_canonical_event_world_model.py",
        "verify_robotwin2_move_can_pot_public_materialization_v1.py",
        "robotwin2_move_can_pot_analytic_event_spec_v1.py",
    )
    if paths.code_root.is_symlink() or not paths.code_root.is_dir():
        raise PairedWatcherError("deployed code root is missing or symbolic")
    if paths.code_root.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PairedWatcherError("deployed code root must be read-only")
    if EXPECTED_RUNNER_SHA256 is None or EXPECTED_EVALUATOR_SHA256 is None:
        raise PairedWatcherError(
            "formal runner/evaluator SHA-256 identities are not frozen; deployment blocked"
        )
    rows = []
    for name in names:
        path = paths.code_root / name
        if path.is_symlink() or not path.is_file():
            raise PairedWatcherError(f"deployed code file is missing or symbolic: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise PairedWatcherError(f"deployed code file must be read-only: {path}")
        rows.append({"path": str(path), "sha256": sha256_file(path)})
    observed = {Path(row["path"]).name: row["sha256"] for row in rows}
    if observed[paths.runner.name] != EXPECTED_RUNNER_SHA256:
        raise PairedWatcherError("deployed formal runner SHA-256 changed")
    if observed[paths.evaluator.name] != EXPECTED_EVALUATOR_SHA256:
        raise PairedWatcherError("deployed formal evaluator SHA-256 changed")
    if observed[paths.event_module.name] != EXPECTED_EVENT_MODULE_SHA256:
        raise PairedWatcherError("deployed analytic event module SHA-256 changed")
    return rows


def validate_materialization_receipt(path: Path) -> dict[str, Any]:
    value = read_json(path, "public materialization v1 receipt")
    if (
        sha256_file(path) != EXPECTED_MATERIALIZATION_RECEIPT_FILE_SHA256
        or value.get("format") != MATERIALIZATION_FORMAT
        or value.get("status") != MATERIALIZATION_STATUS
        or value.get("preregistration_sha256")
        != EXPECTED_MATERIALIZATION_PREREGISTRATION_SHA256
        or value.get("materialized") is not True
        or value.get("official_file_count") != 11
        or value.get("all_exact_archive_payload_sha256_verified") is not True
    ):
        raise PairedWatcherError("public source-slice materialization v1 changed")
    unsigned = dict(value)
    logical = unsigned.pop("materialization_receipt_sha256", None)
    if logical != canonical_sha256(unsigned):
        raise PairedWatcherError("materialization v1 canonical SHA-256 changed")
    return {
        "role": "public_source_slice_materialization_only_not_paired_metrics",
        "path": str(path.resolve()),
        "file_sha256": EXPECTED_MATERIALIZATION_RECEIPT_FILE_SHA256,
        "materialization_receipt_sha256": logical,
        "source_slice_preregistration_sha256": (
            EXPECTED_MATERIALIZATION_PREREGISTRATION_SHA256
        ),
    }


def validate_metrics_preregistration(path: Path) -> dict[str, Any]:
    value = read_json(path, "paired metrics preregistration v2")
    unsigned = dict(value)
    logical = unsigned.pop("preregistration_sha256", None)
    if (
        value.get("format") != METRICS_PREREGISTRATION_FORMAT
        or value.get("status") != METRICS_PREREGISTRATION_STATUS
        or logical != EXPECTED_METRICS_PREREGISTRATION_SHA256
        or logical != canonical_sha256(unsigned)
        or value.get("paired_evaluation_protocol", {}).get("paired_trial_count")
        != EXPECTED_PAIRS
        or value.get("paired_evaluation_protocol", {}).get("planned_rollout_count")
        != EXPECTED_ROLLOUTS
        or value.get("metrics_protocol", {}).get("format")
        != "etsf_robotwin2_move_can_pot_crossbody_metrics_v2"
    ):
        raise PairedWatcherError("paired metrics preregistration v2 changed")
    return {
        "role": "prospective_paired_execution_and_metrics_not_source_materialization",
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "preregistration_sha256": logical,
    }


def validate_static_inputs(paths: FormalPaths) -> dict[str, Any]:
    for path, label in (
        (paths.runner_python, "runner Python"),
        (paths.evaluator_python, "evaluator Python"),
        (paths.robotwin_root, "RoboTwin root"),
        (paths.vlm_metadata, "VLM metadata"),
        (paths.lerobot_root, "LeRobot root"),
        (paths.lerobot_site, "LeRobot site-packages"),
        (paths.robotwin_eval_site, "RoboTwin evaluation site-packages"),
    ):
        if not path.exists():
            raise PairedWatcherError(f"fixed {label} is missing: {path}")
    if sha256_file(paths.event_spec) != EXPECTED_EVENT_SPEC_SHA256:
        raise PairedWatcherError("fixed event spec SHA-256 changed")
    return {
        "code_files": code_binding(paths),
        "event_spec": {
            "path": str(paths.event_spec),
            "file_sha256": EXPECTED_EVENT_SPEC_SHA256,
            "derivation_module": str(paths.event_module),
            "derivation_module_file_sha256": EXPECTED_EVENT_MODULE_SHA256,
        },
        "materialization_v1": validate_materialization_receipt(
            paths.materialization_receipt
        ),
        "metrics_preregistration_v2": validate_metrics_preregistration(
            paths.metrics_preregistration
        ),
    }


def query_gpu(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    identity = run(
        [
            "nvidia-smi",
            f"--id={EXPECTED_GPU_INDEX}",
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if identity.returncode != 0:
        raise PairedWatcherError(f"GPU identity query failed: {identity.stderr.strip()}")
    rows = [row.strip() for row in identity.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise PairedWatcherError("expected exactly one GPU identity row")
    fields = [field.strip() for field in rows[0].split(",", 2)]
    if (
        len(fields) != 3
        or fields[0] != str(EXPECTED_GPU_INDEX)
        or "4090" not in fields[1]
        or fields[2] != EXPECTED_GPU_UUID
    ):
        raise PairedWatcherError(f"unexpected GPU authority: {fields}")
    processes = run(
        [
            "nvidia-smi",
            f"--id={EXPECTED_GPU_INDEX}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if processes.returncode != 0:
        raise PairedWatcherError(
            f"GPU compute-process query failed: {processes.stderr.strip()}"
        )
    pids: list[int] = []
    for raw in processes.stdout.splitlines():
        value = raw.strip()
        if not value or value in {"[N/A]", "N/A", "No running processes found"}:
            continue
        if not value.isdigit() or int(value) <= 0:
            raise PairedWatcherError(f"unrecognized GPU compute PID row: {value!r}")
        pids.append(int(value))
    return {
        "index": EXPECTED_GPU_INDEX,
        "name": fields[1],
        "uuid": fields[2],
        "compute_pids": sorted(set(pids)),
    }


def wait_for_idle_gpu(
    *,
    poll_seconds: float,
    confirmation_seconds: float,
    state_writer: Callable[..., None],
    gpu_reader: Callable[[], dict[str, Any]] = query_gpu,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    while True:
        first = gpu_reader()
        if first["compute_pids"]:
            state_writer(
                "waiting_for_authorized_idle_rtx4090",
                gpu=first,
                gpu_reserved_by_watcher=False,
            )
            sleep(poll_seconds)
            continue
        state_writer(
            "confirming_authorized_idle_rtx4090",
            first_idle_audit=first,
            confirmation_seconds=confirmation_seconds,
            gpu_reserved_by_watcher=False,
        )
        sleep(confirmation_seconds)
        second = gpu_reader()
        if not second["compute_pids"] and second == first:
            return [first, second]
        state_writer(
            "waiting_for_authorized_idle_rtx4090",
            first_idle_audit=first,
            second_audit=second,
            gpu_reserved_by_watcher=False,
        )
        sleep(poll_seconds)


def build_runner_command(paths: FormalPaths) -> list[str]:
    command = [
        str(paths.runner_python),
        str(paths.runner),
        "--actor-checkpoint",
        str(paths.actor_checkpoint),
        "--vlm-metadata-path",
        str(paths.vlm_metadata),
        "--robotwin-root",
        str(paths.robotwin_root),
        "--event-spec",
        str(paths.event_spec),
        "--preregistration",
        str(paths.metrics_preregistration),
    ]
    for body in BODIES:
        command.extend(("--lobo-fold", f"{body}={paths.fold_root(body)}"))
    command.extend(
        (
            "--output",
            str(paths.output_root),
            "--action-exec-steps",
            str(ACTION_EXEC_STEPS),
            "--max-steps",
            str(MAX_STEPS),
            "--fps",
            str(FPS),
        )
    )
    return command


def build_evaluator_command(paths: FormalPaths, outcomes_sha256: str) -> list[str]:
    return [
        str(paths.evaluator_python),
        str(paths.evaluator),
        "--input",
        str(paths.outcomes),
        "--input-file-sha256",
        outcomes_sha256,
        "--output",
        str(paths.report),
    ]


def runner_environment(paths: FormalPaths) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(EXPECTED_GPU_INDEX),
            "PYTHONPATH": ":".join(
                (
                    str(paths.code_root),
                    str(paths.lerobot_root / "src"),
                    str(paths.lerobot_site),
                    str(paths.robotwin_eval_site),
                    str(paths.robotwin_root),
                    str(paths.robotwin_root / "envs/curobo/src"),
                )
            ),
            "ASSETS_PATH": str(paths.robotwin_root),
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return environment


def build_execution_binding(
    paths: FormalPaths,
    static: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    base = {
        "format": BINDING_FORMAT,
        "status": "frozen_before_paired_simulator_execution",
        "pair_count": EXPECTED_PAIRS,
        "rollout_count": EXPECTED_ROLLOUTS,
        "actor_checkpoint": sha256_tree(paths.actor_checkpoint, "actor checkpoint"),
        "vlm_metadata": sha256_tree(paths.vlm_metadata, "VLM metadata"),
        "robotwin_root": str(paths.robotwin_root.resolve(strict=True)),
        "event_spec": static["event_spec"],
        "materialization_v1": static["materialization_v1"],
        "metrics_preregistration_v2": static["metrics_preregistration_v2"],
        "materialization_and_metrics_preregistration_are_distinct_authorities": True,
        "folds": upstream["folds"],
        "upstream": {
            key: value for key, value in upstream.items() if key != "folds"
        },
        "code_files": static["code_files"],
        "runner_command": build_runner_command(paths),
        "evaluator": str(paths.evaluator),
        "gpu_authority": {
            "index": EXPECTED_GPU_INDEX,
            "uuid": EXPECTED_GPU_UUID,
            "name_must_contain": "4090",
            "launch_only_after_two_consecutive_idle_audits": True,
        },
        "training_authorized": False,
        "paired_simulator_execution_authorized": True,
        "promotion_or_deployment_authorized": False,
    }
    return {**base, "logical_sha256": canonical_sha256(base)}


def validate_runner_contract(paths: FormalPaths, binding: Mapping[str, Any]) -> None:
    contract = read_json(paths.execution_contract, "runner execution contract")
    if (
        contract.get("pair_count") != EXPECTED_PAIRS
        or contract.get("rollout_count") != EXPECTED_ROLLOUTS
        or contract.get("actor_checkpoint") != str(paths.actor_checkpoint)
        or contract.get("actor_checkpoint_tree_sha256")
        != binding["actor_checkpoint"]["tree_sha256"]
        or contract.get("vlm_metadata_path") != str(paths.vlm_metadata)
        or contract.get("vlm_metadata_tree_sha256")
        != binding["vlm_metadata"]["tree_sha256"]
        or contract.get("event_spec") != str(paths.event_spec)
        or contract.get("event_spec_sha256") != EXPECTED_EVENT_SPEC_SHA256
        or contract.get("event_derivation_implementation_sha256")
        != EXPECTED_EVENT_MODULE_SHA256
        or contract.get("training_and_online_event_implementation_identical") is not True
        or contract.get("preregistration") != str(paths.metrics_preregistration)
        or contract.get("preregistration_sha256")
        != EXPECTED_METRICS_PREREGISTRATION_SHA256
        or contract.get("action_exec_steps") != ACTION_EXEC_STEPS
        or contract.get("max_steps") != MAX_STEPS
        or contract.get("fps") != float(FPS)
        or contract.get("candidate_rank_ensemble_contract")
        != STANDARDIZED_RANK_ENSEMBLE_CONTRACT
        or contract.get("no_training") is not True
    ):
        raise PairedWatcherError("runner execution contract differs from watcher binding")
    contract_folds = contract.get("folds")
    if not isinstance(contract_folds, Mapping) or set(contract_folds) != set(BODIES):
        raise PairedWatcherError("runner execution contract fold set changed")
    for body in BODIES:
        if (
            contract_folds[body].get("training_summary_sha256")
            != binding["folds"][body]["training_summary_sha256"]
            or contract_folds[body].get("members")
            != binding["folds"][body]["members"]
        ):
            raise PairedWatcherError(f"runner execution contract changed fold {body}")


def validate_report(paths: FormalPaths, outcomes_sha256: str) -> dict[str, Any]:
    report = read_json(paths.report, "paired success report")
    unsigned = dict(report)
    reported_sha = unsigned.pop("report_sha256", None)
    if (
        report.get("format") != REPORT_FORMAT
        or report.get("status") != REPORT_STATUS
        or report.get("pair_count") != EXPECTED_PAIRS
        or report.get("planned_rollout_count") != EXPECTED_ROLLOUTS
        or report.get("input_binding", {}).get("input_file_sha256")
        != outcomes_sha256
        or reported_sha != canonical_sha256(unsigned)
    ):
        raise PairedWatcherError("paired success report contract or SHA changed")
    return report


def validate_completion_receipt(paths: FormalPaths) -> dict[str, Any]:
    if not paths.completion_receipt.is_file() or paths.completion_receipt.is_symlink():
        raise PairedWatcherError("paired runner did not create a real completion receipt")
    receipt = read_json(paths.completion_receipt, "paired execution completion receipt")
    receipt_unsigned = dict(receipt)
    receipt_logical = receipt_unsigned.pop("logical_sha256", None)
    contract = read_json(paths.execution_contract, "runner execution contract")
    outcomes = read_json(paths.outcomes, "paired outcomes")
    outcomes_unsigned = dict(outcomes)
    outcomes_document_sha = outcomes_unsigned.pop("document_sha256", None)
    if (
        receipt.get("format")
        != "etsf_robotwin2_paired_execution_completion_receipt_v1"
        or receipt.get("status") != "complete_1000_pairs_2000_rollouts_frozen"
        or receipt.get("pair_count") != EXPECTED_PAIRS
        or receipt.get("rollout_count") != EXPECTED_ROLLOUTS
        or receipt.get("execution_contract_path") != str(paths.execution_contract)
        or receipt.get("execution_contract_logical_sha256")
        != contract.get("logical_sha256")
        or receipt.get("execution_contract_file_sha256")
        != sha256_file(paths.execution_contract)
        or receipt.get("candidate_rank_ensemble_contract")
        != contract.get("candidate_rank_ensemble_contract")
        or receipt.get("ordered_pair_sha256s_sha256")
        != outcomes.get("ordered_pair_sha256s_sha256")
        or receipt.get("outcome_path") != str(paths.outcomes)
        or receipt.get("outcome_document_sha256") != outcomes_document_sha
        or receipt.get("outcome_file_sha256") != sha256_file(paths.outcomes)
        or outcomes.get("execution_contract_logical_sha256")
        != contract.get("logical_sha256")
        or outcomes.get("execution_contract_file_sha256")
        != sha256_file(paths.execution_contract)
        or outcomes_document_sha != canonical_sha256(outcomes_unsigned)
        or receipt_logical != canonical_sha256(receipt_unsigned)
    ):
        raise PairedWatcherError("paired completion receipt or evidence chain changed")
    return receipt


def acquire_instance_lock(paths: FormalPaths) -> Any:
    paths.instance_lock.parent.mkdir(parents=True, exist_ok=True)
    stream = paths.instance_lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise PairedWatcherError("another formal paired watcher instance is active") from error
    stream.seek(0)
    stream.truncate()
    stream.write(f"{os.getpid()}\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def acquire_gpu_lock(paths: FormalPaths, audits: Sequence[Mapping[str, Any]]) -> None:
    value = {
        "format": "etsf_robotwin2_five_body_paired_success_gpu_reservation_v1",
        "pid": os.getpid(),
        "acquired_at_utc": utc_now(),
        "gpu_idle_audits": list(audits),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(paths.gpu_lock, flags, 0o600)
    except FileExistsError as error:
        raise PairedWatcherError(f"paired GPU reservation already exists: {paths.gpu_lock}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def release_gpu_lock(paths: FormalPaths) -> None:
    if paths.gpu_lock.exists():
        paths.gpu_lock.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--idle-confirmation-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.idle_confirmation_seconds <= 0:
        raise PairedWatcherError("poll and idle-confirmation intervals must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    paths = formal_paths()
    instance_lock = acquire_instance_lock(paths)
    atomic_text(paths.pid, f"{os.getpid()}\n")
    static = validate_static_inputs(paths)

    def write_state(status_value: str, **extra: Any) -> None:
        atomic_json(
            paths.state,
            {
                "format": FORMAT,
                "status": status_value,
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "upstream_state": str(paths.upstream_state),
                "upstream_run_exit": str(paths.upstream_run_exit),
                "lobo_root": str(paths.lobo_root),
                "actor_checkpoint": str(paths.actor_checkpoint),
                "vlm_metadata": str(paths.vlm_metadata),
                "robotwin_root": str(paths.robotwin_root),
                "event_spec": str(paths.event_spec),
                "event_derivation_module": str(paths.event_module),
                "materialization_v1": str(paths.materialization_receipt),
                "metrics_preregistration_v2": str(paths.metrics_preregistration),
                "output_root": str(paths.output_root),
                "watcher_log": str(paths.watcher_log),
                "expected_pairs": EXPECTED_PAIRS,
                "expected_rollouts": EXPECTED_ROLLOUTS,
                **extra,
            },
        )

    upstream: dict[str, Any] | None = None
    while upstream is None:
        upstream, progress = probe_upstream(paths)
        if upstream is None:
            write_state(
                "waiting_for_true_complete_five_fold_lobo",
                upstream_progress=progress,
                gpu_reserved_by_watcher=False,
                cuda_or_simulator_imported_by_watcher=False,
            )
            time.sleep(arguments.poll_seconds)

    paths.output_root.mkdir(parents=True, exist_ok=True)
    if paths.output_root.is_symlink() or not paths.output_root.is_dir():
        raise PairedWatcherError("paired output root is symbolic or not a directory")
    binding = build_execution_binding(paths, static, upstream)
    create_once_or_verify(paths.execution_binding, binding, "watcher execution binding")
    paths.runner_log.parent.mkdir(parents=True, exist_ok=True)

    if not paths.outcomes.exists():
        audits = wait_for_idle_gpu(
            poll_seconds=arguments.poll_seconds,
            confirmation_seconds=arguments.idle_confirmation_seconds,
            state_writer=write_state,
        )
        acquire_gpu_lock(paths, audits)
        try:
            final_audit = query_gpu()
            if final_audit["compute_pids"]:
                raise PairedWatcherError(
                    f"RTX 4090 became busy before runner launch: {final_audit['compute_pids']}"
                )
            command = build_runner_command(paths)
            write_state(
                "running_1000_pairs_2000_rollouts",
                upstream_binding=upstream,
                execution_binding=str(paths.execution_binding),
                execution_binding_logical_sha256=binding["logical_sha256"],
                runner_command=command,
                runner_log=str(paths.runner_log),
                gpu_idle_audits=audits,
                launch_gpu_audit=final_audit,
                gpu_reserved_by_watcher=True,
            )
            with paths.runner_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    "\nWATCHER_RUNNER_INVOCATION="
                    + json.dumps(command, ensure_ascii=True)
                    + "\n"
                )
                stream.flush()
                result = subprocess.run(
                    command,
                    cwd=paths.robotwin_root,
                    env=runner_environment(paths),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise PairedWatcherError(f"paired runner failed with exit {result.returncode}")
        finally:
            release_gpu_lock(paths)

    if not paths.outcomes.is_file() or paths.outcomes.is_symlink():
        raise PairedWatcherError("paired runner did not create a real paired_outcomes.json")
    validate_runner_contract(paths, binding)
    outcomes_sha = sha256_file(paths.outcomes)
    completion_receipt = validate_completion_receipt(paths)
    if not paths.report.exists():
        evaluator_command = build_evaluator_command(paths, outcomes_sha)
        write_state(
            "evaluating_complete_paired_outcomes",
            execution_binding=str(paths.execution_binding),
            execution_binding_logical_sha256=binding["logical_sha256"],
            paired_outcomes=str(paths.outcomes),
            paired_outcomes_file_sha256=outcomes_sha,
            evaluator_command=evaluator_command,
            evaluator_log=str(paths.evaluator_log),
            gpu_reserved_by_watcher=False,
        )
        evaluator_environment = os.environ.copy()
        evaluator_environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(paths.code_root),
            }
        )
        with paths.evaluator_log.open("a", encoding="utf-8") as stream:
            stream.write(
                "\nWATCHER_EVALUATOR_INVOCATION="
                + json.dumps(evaluator_command, ensure_ascii=True)
                + "\n"
            )
            stream.flush()
            evaluated = subprocess.run(
                evaluator_command,
                cwd=paths.code_root,
                env=evaluator_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if evaluated.returncode != 0:
            raise PairedWatcherError(f"paired evaluator failed with exit {evaluated.returncode}")
    report = validate_report(paths, outcomes_sha)
    atomic_text(paths.run_exit, "0\n")
    write_state(
        "complete",
        execution_binding=str(paths.execution_binding),
        execution_binding_file_sha256=sha256_file(paths.execution_binding),
        execution_binding_logical_sha256=binding["logical_sha256"],
        paired_outcomes=str(paths.outcomes),
        paired_outcomes_file_sha256=outcomes_sha,
        paired_completion_receipt=str(paths.completion_receipt),
        paired_completion_receipt_file_sha256=sha256_file(
            paths.completion_receipt
        ),
        ordered_pair_sha256s_sha256=completion_receipt[
            "ordered_pair_sha256s_sha256"
        ],
        paired_success_report=str(paths.report),
        paired_success_report_file_sha256=sha256_file(paths.report),
        paired_success_report_logical_sha256=report["report_sha256"],
        completed_pairs=report["pair_count"],
        completed_rollouts=report["planned_rollout_count"],
        prospective_improvement_gate_passed=report["prospective_improvement_gate"][
            "passed"
        ],
        gpu_reserved_by_watcher=False,
        promotion_or_deployment_authorized=False,
    )
    # Keep the advisory lock stream alive until all terminal state is durable.
    instance_lock.flush()
    return 0


def record_failure(error: BaseException) -> None:
    paths = formal_paths()
    release_gpu_lock(paths)
    try:
        atomic_text(paths.run_exit, "1\n")
        atomic_json(
            paths.state,
            {
                "format": FORMAT,
                "status": "failed",
                "updated_at_utc": utc_now(),
                "pid": os.getpid(),
                "error": f"{type(error).__name__}: {error}",
                "upstream_state": str(paths.upstream_state),
                "output_root": str(paths.output_root),
                "watcher_log": str(paths.watcher_log),
                "gpu_reserved_by_watcher": False,
            },
        )
    except Exception as state_error:
        print(
            f"failure-state write also failed: {type(state_error).__name__}: {state_error}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as error:
        record_failure(error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    raise SystemExit(exit_code)


__all__ = [
    "BODIES",
    "FormalPaths",
    "PairedWatcherError",
    "build_evaluator_command",
    "build_runner_command",
    "canonical_sha256",
    "formal_paths",
    "inspect_fold",
    "probe_upstream",
    "query_gpu",
    "validate_report",
    "validate_upstream_complete",
    "wait_for_idle_gpu",
]
