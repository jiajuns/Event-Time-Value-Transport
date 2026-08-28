#!/usr/bin/env python3
"""Detached fail-closed watcher for SmolVLA schema-v5 source63 training.

The watcher deliberately has a hard two-phase boundary.  Before the collector
publishes a materialized ``run.exit`` containing exactly ``0``, it reads only
that sentinel and writes launcher heartbeat JSON.  It does not read the source
manifest and cannot import or invoke the HDF5 group validator.  After the zero
exit, it authenticates the complete 63-group manifest against the frozen split,
self-validates only the 44 train and 14 validation HDF5 groups, byte-hashes the
five label-sealed development holdouts without opening their HDF5 datasets,
copies an independent read-only snapshot, initializes the native dual-reservation
core, and launches the counterfactual trainer on one audited RTX 4090.

No path containing ``fresh`` or ``confirmation`` is accepted.  Output roots are
create-once and non-resumable.  ``detach`` starts ``run`` in a new OS session so
the server-side watcher and training survive an SSH client disconnect.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping, Sequence


FORMAT = "etsf_smolvla_schema5_source63_native_training_launcher_v1"
STATE_FORMAT = "etsf_smolvla_schema5_source63_native_training_state_v1"
SNAPSHOT_FORMAT = "etsf_smolvla_schema5_source63_snapshot_v1"
DETACH_FORMAT = "etsf_smolvla_schema5_source63_detach_receipt_v1"
EXTERNAL_SUITE_GUARD_FORMAT = "etsf_external_suite_parent_identity_guard_v1"
GPU_IDLE_AUDIT_FORMAT = "etsf_guarded_rtx4090_idle_release_audit_v1"
GPU_LOCK_RELEASE_AUDIT_FORMAT = "etsf_owned_gpu_lock_release_audit_v1"
TREE_FREEZE_CONTRACT_FORMAT = "etsf_source63_terminal_tree_freeze_contract_v1"
TERMINAL_RECEIPT_NAMES = frozenset({"final_receipt.json", "failure_receipt.json"})
TERMINAL_STATUS = "complete_source63_native_counterfactual_training_fresh_forbidden"
FAILURE_STATUS = "failed_closed_source63_native_counterfactual_training_fresh_forbidden"
FAILURE_PHASES = frozenset(
    {
        "before_gpu_idle_guard",
        "gpu_idle_guard_running",
        "after_gpu_idle_release_before_training",
        "training_runtime_probe",
        "training_pre_popen_guard",
        "training_popen_attempt",
        "training_process_started",
        "after_training_stage_before_gpu_lock_release",
        "after_gpu_lock_release_before_terminal_finalize",
    }
)
EXPECTED_GROUPS = 63
EXPECTED_SCHEMA = 5
EXPECTED_TASK = "move_can_pot"
EXPECTED_BODY = "aloha-agilex"
EXPECTED_POLICY = "smolvla"
EXPECTED_CANDIDATES = 4
EXPECTED_HIDDEN_DIM = 960
EXPECTED_ACTION_DIM = 14
EXPECTED_ACTION_CHUNK = 50
TRAINING_SEEDS = (20260828, 20260829, 20260830, 20260831, 20260832)
TRAINING_STEPS = 3000
INITIALIZATION_SEED = 20260828
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
SHA256_CHARS = frozenset("0123456789abcdef")
ENTRYPOINTS = (
    "launch_smolvla_schema5_source63_native_training.py",
    "etsf_torch_weights_only_compat_v1.py",
    "run_etsf_bound_python_stage.py",
    "initialize_smolvla_schema5_native_event_core.py",
    "train_openvla_etsf_counterfactual.py",
    "collect_smolvla_etsf_event_branches.py",
    "etsf_policy_feature_action_bridge.py",
    "example_smolvla_event_critic_adapter.py",
    "example_openvla_event_critic_plugin.py",
    "openvla_etsf_event_critic_plugin.py",
)
RUNTIME_CONTRACT_FORMAT = "etsf_isolated_python_torch_runtime_v1"
ENVIRONMENT_CONTRACT_FORMAT = "etsf_canonical_python_environment_v1"
FORCED_PYTHON_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
}
SCRUBBED_ENVIRONMENT_NAMES = frozenset(
    {
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PYTHON_EXE",
        "__PYVENV_LAUNCHER__",
    }
)

RUNTIME_PROBE_SOURCE = r"""
import hashlib
import json
import pathlib
import sys

import torch

def _file(path_value):
    path = pathlib.Path(path_value).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}

value = {
    "format": "etsf_isolated_python_torch_runtime_v1",
    "isolated": bool(sys.flags.isolated),
    "python_executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
    "python_version": str(sys.version),
    "python_prefix": str(pathlib.Path(sys.prefix).resolve(strict=True)),
    "python_base_prefix": str(pathlib.Path(sys.base_prefix).resolve(strict=True)),
    "torch_version": str(torch.__version__),
    "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
    "torch_module": _file(torch.__file__),
    "torch_c_module": _file(torch._C.__file__),
}
payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
value["runtime_contract_sha256"] = hashlib.sha256(payload).hexdigest()
print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
"""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sign_canonical_receipt(
    value: Mapping[str, Any], *, field: str = "receipt_sha256"
) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise ValueError(f"receipt already contains {field}")
    result[field] = canonical_sha256(result)
    return result


def verify_canonical_receipt(
    value: Mapping[str, Any], *, field: str = "receipt_sha256"
) -> dict[str, Any]:
    result = dict(value)
    claimed = result.pop(field, None)
    if not _is_sha256(claimed) or claimed != canonical_sha256(result):
        raise RuntimeError("canonical receipt SHA256 is invalid")
    return dict(value)


def _verify_embedded_canonical_sha(
    value: Mapping[str, Any], field: str, *, role: str
) -> dict[str, Any]:
    result = dict(value)
    claimed = result.pop(field, None)
    if not _is_sha256(claimed) or claimed != canonical_sha256(result):
        raise RuntimeError(f"{role} SHA256 is invalid")
    return dict(value)


def _validate_gpu_idle_audit_semantics(value: Mapping[str, Any]) -> None:
    _verify_embedded_canonical_sha(
        value,
        "gpu_idle_release_audit_sha256",
        role="GPU idle release audit",
    )
    status = value.get("status")
    checks = value.get("checks")
    idle_confirmations = value.get("idle_confirmations")
    observations = value.get("valid_idle_observations")
    gpu_index = value.get("gpu_index")
    gpu_name = value.get("gpu_name")
    gpu_uuid = value.get("gpu_uuid")
    if (
        value.get("format") != GPU_IDLE_AUDIT_FORMAT
        or status
        not in (
            "waiting_for_guarded_rtx4090_release",
            "complete_two_idle_samples_released_for_training",
        )
        or isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
        or not isinstance(gpu_name, str)
        or "4090" not in gpu_name
        or not isinstance(gpu_uuid, str)
        or not gpu_uuid
        or isinstance(checks, bool)
        or not isinstance(checks, int)
        or checks < 0
        or not isinstance(value.get("wait_seconds"), (int, float))
        or value.get("wait_seconds") < 0
        or not isinstance(value.get("external_suite_parent_guard_enabled"), bool)
        or value.get("idle_confirmations_required") != 2
        or isinstance(idle_confirmations, bool)
        or not isinstance(idle_confirmations, int)
        or not isinstance(value.get("external_suite_parent_gone_latched"), bool)
        or not _is_sha256(value.get("observation_chain_sha256"))
        or not isinstance(observations, list)
    ):
        raise RuntimeError("GPU idle release audit semantics are incomplete")
    previous_observed_unix: float | None = None
    for observation in observations:
        if (
            not isinstance(observation, Mapping)
            or isinstance(observation.get("check"), bool)
            or not isinstance(observation.get("check"), int)
            or not 1 <= observation.get("check") <= checks
            or observation.get("parent_alive_before_gpu_query") is not False
            or observation.get("parent_alive_after_gpu_query") is not False
            or observation.get("compute_pids") != []
            or observation.get("gpu_index") != gpu_index
            or observation.get("gpu_name") != gpu_name
            or observation.get("gpu_uuid") != gpu_uuid
            or not isinstance(observation.get("observed_unix"), (int, float))
            or not _is_sha256(observation.get("observation_chain_sha256"))
        ):
            raise RuntimeError("GPU idle release observation semantics are incomplete")
        observed_unix = float(observation["observed_unix"])
        if previous_observed_unix is not None and observed_unix < previous_observed_unix:
            raise RuntimeError("GPU idle release observations are out of order")
        previous_observed_unix = observed_unix
    if observations and (
        observations[-1].get("observation_chain_sha256")
        != value.get("observation_chain_sha256")
    ):
        raise RuntimeError("GPU idle release observation chain is inconsistent")
    if status == "complete_two_idle_samples_released_for_training":
        if (
            checks < 2
            or idle_confirmations != 2
            or len(observations) != 2
            or [observation.get("check") for observation in observations]
            != [checks - 1, checks]
            or value.get("compute_pids") != []
            or value.get("external_suite_parent_alive") is not False
            or value.get("external_suite_parent_alive_before_gpu_query") is not False
            or value.get("external_suite_parent_alive_after_gpu_query") is not False
            or value.get("external_suite_parent_gone_latched") is not True
        ):
            raise RuntimeError("GPU idle release completion semantics are incomplete")
    else:
        compute_pids = value.get("compute_pids")
        if (
            idle_confirmations not in (0, 1)
            or len(observations) != idle_confirmations
            or checks < idle_confirmations
            or (
                len(observations) == 1
                and observations[0].get("check") != checks
            )
            or (
                checks == 0
                and (
                    compute_pids is not None
                    or value.get("external_suite_parent_alive") is not None
                    or value.get("external_suite_parent_alive_before_gpu_query") is not None
                    or value.get("external_suite_parent_alive_after_gpu_query") is not None
                )
            )
            or (
                checks > 0
                and (
                    not isinstance(compute_pids, list)
                    or any(
                        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                        for pid in compute_pids
                    )
                    or not isinstance(value.get("external_suite_parent_alive"), bool)
                    or not isinstance(
                        value.get("external_suite_parent_alive_before_gpu_query"), bool
                    )
                    or not isinstance(
                        value.get("external_suite_parent_alive_after_gpu_query"), bool
                    )
                )
            )
        ):
            raise RuntimeError("GPU idle waiting semantics are incomplete")


def _validate_gpu_lock_release_audit_semantics(value: Mapping[str, Any]) -> None:
    _verify_embedded_canonical_sha(
        value,
        "release_audit_sha256",
        role="GPU lock release audit",
    )
    path = value.get("path")
    pid = value.get("pid")
    released = value.get("released")
    if (
        value.get("format") != GPU_LOCK_RELEASE_AUDIT_FORMAT
        or not isinstance(path, str)
        or not Path(path).is_absolute()
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not _is_sha256(value.get("token_sha256"))
        or not isinstance(released, bool)
        or not isinstance(value.get("observed_unix"), (int, float))
    ):
        raise RuntimeError("GPU lock release audit semantics are incomplete")
    if released:
        if (
            value.get("status") != "released_exact_owned_gpu_lock"
            or value.get("observed_owner_pid") != pid
            or value.get("observed_token_sha256") != value.get("token_sha256")
            or not isinstance(value.get("released_unix"), (int, float))
            or value.get("released_unix") < value.get("observed_unix")
            or "error" in value
            or "error_type" in value
        ):
            raise RuntimeError("GPU lock release success semantics are incomplete")
    elif (
        value.get("status") != "release_failed_closed"
        or not isinstance(value.get("error_type"), str)
        or not value.get("error_type")
        or not isinstance(value.get("error"), str)
        or not value.get("error")
        or "released_unix" in value
    ):
        raise RuntimeError("GPU lock release failure semantics are incomplete")


def validate_source63_terminal_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = verify_canonical_receipt(value)
    if receipt.get("format") != FORMAT or receipt.get("status") not in (
        TERMINAL_STATUS,
        FAILURE_STATUS,
    ):
        raise RuntimeError("source63 terminal receipt status is invalid")
    launcher_pid = receipt.get("launcher_pid")
    gpu_lock_path = receipt.get("gpu_lock_path")
    gpu_lock_token_sha256 = receipt.get("gpu_lock_token_sha256")
    terminal_gpu_identity = receipt.get("gpu_identity")
    output_root = receipt.get("output_root")
    expected_terminal_name = (
        "final_receipt.json"
        if receipt.get("status") == TERMINAL_STATUS
        else "failure_receipt.json"
    )
    artifacts_frozen = receipt.get("artifacts_frozen_read_only")
    freeze_contract = receipt.get("artifact_freeze_contract")
    if (
        isinstance(launcher_pid, bool)
        or not isinstance(launcher_pid, int)
        or launcher_pid <= 0
        or not isinstance(gpu_lock_path, str)
        or not Path(gpu_lock_path).is_absolute()
        or not _is_sha256(gpu_lock_token_sha256)
        or not isinstance(terminal_gpu_identity, Mapping)
        or set(terminal_gpu_identity) != {"gpu_index", "gpu_name", "gpu_uuid"}
        or isinstance(terminal_gpu_identity.get("gpu_index"), bool)
        or not isinstance(terminal_gpu_identity.get("gpu_index"), int)
        or terminal_gpu_identity.get("gpu_index") < 0
        or not isinstance(terminal_gpu_identity.get("gpu_name"), str)
        or "4090" not in terminal_gpu_identity.get("gpu_name")
        or not isinstance(terminal_gpu_identity.get("gpu_uuid"), str)
        or not terminal_gpu_identity.get("gpu_uuid")
        or not isinstance(output_root, str)
        or not Path(output_root).is_absolute()
        or receipt.get("terminal_receipt_name") != expected_terminal_name
        or not isinstance(artifacts_frozen, bool)
    ):
        raise RuntimeError("source63 terminal GPU lock identity is invalid")
    if artifacts_frozen:
        if not isinstance(freeze_contract, Mapping):
            raise RuntimeError("source63 frozen terminal lacks its tree contract")
        _validate_tree_freeze_contract_semantics(freeze_contract)
        if (
            freeze_contract.get("output_root") != output_root
            or receipt.get("artifact_freeze_contract_sha256")
            != freeze_contract.get("tree_freeze_contract_sha256")
        ):
            raise RuntimeError("source63 terminal tree freeze binding changed")
    elif (
        freeze_contract is not None
        or receipt.get("artifact_freeze_contract_sha256") is not None
    ):
        raise RuntimeError("source63 unfrozen terminal has a tree freeze claim")
    guard = receipt.get("external_suite_parent_guard")
    if guard is not None:
        if not isinstance(guard, Mapping):
            raise RuntimeError("source63 terminal guard is invalid")
        if guard.get("enabled") is True:
            _verify_embedded_canonical_sha(
                guard, "guard_contract_sha256", role="external parent guard"
            )
            if receipt.get("external_suite_parent_guard_contract_sha256") != guard.get(
                "guard_contract_sha256"
            ):
                raise RuntimeError("source63 terminal guard binding changed")
    if receipt["status"] == TERMINAL_STATUS:
        idle = receipt.get("gpu_idle_audit")
        start = receipt.get("training_start_guard_audit")
        release = receipt.get("gpu_lock_release_audit")
        stage = receipt.get("training_stage_receipt")
        if not all(
            isinstance(item, Mapping) for item in (idle, start, release, stage)
        ):
            raise RuntimeError("source63 success terminal audits are missing")
        _validate_gpu_idle_audit_semantics(idle)
        _verify_embedded_canonical_sha(
            start, "training_start_guard_audit_sha256", role="training start guard audit"
        )
        _validate_gpu_lock_release_audit_semantics(release)
        if (
            release.get("path") != gpu_lock_path
            or release.get("pid") != launcher_pid
            or release.get("token_sha256") != gpu_lock_token_sha256
        ):
            raise RuntimeError("source63 success GPU lock release binding changed")
        _verify_embedded_canonical_sha(
            stage, "stage_receipt_sha256", role="training stage receipt"
        )
        observations = idle.get("valid_idle_observations")
        idle_gpu_identity = {
            key: idle.get(key) for key in ("gpu_index", "gpu_name", "gpu_uuid")
        }
        if not isinstance(observations, list) or len(observations) != 2:
            raise RuntimeError("source63 success idle observations are incomplete")
        for observation in observations:
            if (
                not isinstance(observation, Mapping)
                or observation.get("parent_alive_before_gpu_query") is not False
                or observation.get("parent_alive_after_gpu_query") is not False
                or observation.get("compute_pids") != []
                or any(
                    observation.get(key) != idle_gpu_identity[key]
                    for key in ("gpu_index", "gpu_name", "gpu_uuid")
                )
                or not isinstance(observation.get("observed_unix"), (int, float))
            ):
                raise RuntimeError("source63 success idle observation semantics changed")
        if observations[1]["observed_unix"] < observations[0]["observed_unix"]:
            raise RuntimeError("source63 success idle observation time order changed")
        stage_sha = stage.get("stage_receipt_sha256")
        idle_sha = idle.get("gpu_idle_release_audit_sha256")
        start_sha = start.get("training_start_guard_audit_sha256")
        release_sha = release.get("release_audit_sha256")
        timeline = (
            observations[1]["observed_unix"],
            start.get("observed_unix"),
            stage.get("popen_unix"),
            stage.get("finished_unix"),
            release.get("released_unix"),
        )
        if (
            idle.get("status") != "complete_two_idle_samples_released_for_training"
            or idle_gpu_identity != terminal_gpu_identity
            or idle.get("external_suite_parent_guard_enabled")
            is not (isinstance(guard, Mapping) and guard.get("enabled") is True)
            or idle.get("external_suite_parent_pid")
            != (guard.get("pid") if isinstance(guard, Mapping) else None)
            or idle.get("external_suite_parent_alive_before_gpu_query") is not False
            or idle.get("external_suite_parent_alive_after_gpu_query") is not False
            or idle.get("external_suite_parent_gone_latched") is not True
            or idle.get("compute_pids") != []
            or idle.get("idle_confirmations") != 2
            or idle.get("idle_confirmations_required") != 2
            or start.get("status")
            != "complete_parent_absent_gpu_idle_script_unchanged"
            or start.get("compute_pids") != []
            or start.get("gpu_identity") != idle_gpu_identity
            or start.get("external_suite_parent_guard_contract_sha256")
            != receipt.get("external_suite_parent_guard_contract_sha256")
            or start.get("gpu_idle_release_audit_sha256")
            != idle_sha
            or receipt.get("gpu_idle_release_audit_sha256")
            != idle_sha
            or receipt.get("training_start_guard_audit_sha256")
            != start_sha
            or stage.get("status") != "complete"
            or stage.get("returncode") != 0
            or isinstance(stage.get("pid"), bool)
            or not isinstance(stage.get("pid"), int)
            or stage.get("process_group_id") != stage.get("pid")
            or stage.get("process_group_isolated") is not True
            or stage.get("process_reaped") is not True
            or stage.get("process_group_reaped") is not True
            or stage.get("pre_popen_guard_audit") != start
            or stage.get("external_suite_parent_guard_contract_sha256")
            != receipt.get("external_suite_parent_guard_contract_sha256")
            or stage.get("gpu_idle_release_audit_sha256") != idle_sha
            or stage.get("training_start_guard_audit_sha256") != start_sha
            or stage.get("gpu_lock_release_audit_sha256") != release_sha
            or receipt.get("training_stage_receipt_sha256") != stage_sha
            or release.get("released") is not True
            or release.get("status") != "released_exact_owned_gpu_lock"
            or receipt.get("gpu_lock_release_audit_sha256")
            != release_sha
            or any(not isinstance(value, (int, float)) for value in timeline)
            or list(timeline) != sorted(timeline)
        ):
            raise RuntimeError("source63 success terminal audit binding is incomplete")
    else:
        partial = receipt.get("partial_gpu_idle_guard_audit")
        guard_started = receipt.get("gpu_idle_guard_started")
        if guard_started is True:
            if not isinstance(partial, Mapping):
                raise RuntimeError("source63 started guard lacks its partial audit")
            _validate_gpu_idle_audit_semantics(partial)
        elif guard_started is False:
            if partial is not None:
                raise RuntimeError("source63 unstarted guard has a partial audit")
        else:
            raise RuntimeError("source63 failure guard-start evidence is invalid")
        release = receipt.get("gpu_lock_release_audit")
        if release is not None:
            if not isinstance(release, Mapping):
                raise RuntimeError("source63 failure lock release audit is invalid")
            _validate_gpu_lock_release_audit_semantics(release)
            if (
                release.get("path") != gpu_lock_path
                or release.get("pid") != launcher_pid
                or release.get("token_sha256") != gpu_lock_token_sha256
            ):
                raise RuntimeError("source63 failure GPU lock release binding changed")
        phase = receipt.get("failure_phase")
        idle_released = receipt.get("gpu_idle_release_reached")
        pre_guard_started = receipt.get("training_pre_popen_guard_started")
        pre_guard_completed = receipt.get("training_pre_popen_guard_completed")
        popen_attempted = receipt.get("training_popen_attempted")
        popen_reached = receipt.get("training_popen_reached")
        process_pid = receipt.get("training_process_pid")
        process_reaped = receipt.get("training_process_reaped")
        process_group_id = receipt.get("training_process_group_id")
        process_group_isolated = receipt.get("training_process_group_isolated")
        process_group_reaped = receipt.get("training_process_group_reaped")
        process_group_binding_status = receipt.get(
            "training_process_group_binding_status"
        )
        stage_returned = receipt.get("training_stage_returned")
        lock_acquired = receipt.get("gpu_lock_acquired")
        lock_released = receipt.get("gpu_lock_released")
        lock_released_before_failure = receipt.get(
            "gpu_lock_released_before_failure"
        )
        unreaped_stage = receipt.get("unreaped_stage_process")
        lock_retained_for_unreaped = receipt.get(
            "gpu_lock_retained_for_unreaped_stage_process"
        )
        booleans = (
            idle_released,
            pre_guard_started,
            pre_guard_completed,
            popen_attempted,
            popen_reached,
            process_reaped,
            process_group_isolated,
            process_group_reaped,
            stage_returned,
            lock_acquired,
            lock_released,
            lock_released_before_failure,
            lock_retained_for_unreaped,
            artifacts_frozen,
        )
        actual_lock_released = (
            isinstance(release, Mapping) and release.get("released") is True
        )
        actual_idle_released = (
            guard_started is True
            and isinstance(partial, Mapping)
            and partial.get("status")
            == "complete_two_idle_samples_released_for_training"
        )
        if guard_started is True and (
            any(
                partial.get(key) != terminal_gpu_identity.get(key)
                for key in ("gpu_index", "gpu_name", "gpu_uuid")
            )
            or partial.get("external_suite_parent_guard_enabled")
            is not (isinstance(guard, Mapping) and guard.get("enabled") is True)
            or partial.get("external_suite_parent_pid")
            != (guard.get("pid") if isinstance(guard, Mapping) else None)
        ):
            raise RuntimeError("source63 failure GPU idle guard binding changed")
        phase_semantics = {
            "before_gpu_idle_guard": (False, False, False, False, False, False),
            "gpu_idle_guard_running": (True, False, False, False, False, False),
            "after_gpu_idle_release_before_training": (
                True,
                True,
                False,
                False,
                False,
                False,
            ),
            "training_runtime_probe": (True, True, False, False, False, False),
            "training_pre_popen_guard": (True, True, True, False, False, False),
            "training_popen_attempt": (True, True, True, True, True, False),
            "training_process_started": (True, True, True, True, True, False),
            "after_training_stage_before_gpu_lock_release": (
                True,
                True,
                True,
                True,
                True,
                True,
            ),
            "after_gpu_lock_release_before_terminal_finalize": (
                True,
                True,
                True,
                True,
                True,
                True,
            ),
        }
        observed_phase_semantics = (
            guard_started,
            idle_released,
            pre_guard_started,
            pre_guard_completed,
            popen_attempted,
            stage_returned,
        )
        if (
            phase not in FAILURE_PHASES
            or any(not isinstance(item, bool) for item in booleans)
            or idle_released is not actual_idle_released
            or lock_released is not actual_lock_released
            or observed_phase_semantics != phase_semantics.get(phase)
            or (idle_released and not guard_started)
            or (pre_guard_started and not idle_released)
            or (pre_guard_completed and not pre_guard_started)
            or (popen_attempted and not pre_guard_completed)
            or (popen_reached and not popen_attempted)
            or (
                process_reaped
                and not popen_reached
                and process_group_binding_status != "popen_attempt_unproven"
            )
            or (process_group_isolated and not popen_reached)
            or (process_group_reaped and not (process_reaped and process_group_isolated))
            or (
                popen_reached
                and (
                    isinstance(process_pid, bool)
                    or not isinstance(process_pid, int)
                    or process_pid <= 0
                )
            )
            or (
                not popen_reached
                and process_group_binding_status == "not_reached"
                and (
                    popen_attempted
                    or process_pid is not None
                    or process_group_id is not None
                    or process_group_isolated
                    or process_group_reaped
                )
            )
            or (
                not popen_reached
                and process_group_binding_status == "popen_attempt_unproven"
                and (
                    not popen_attempted
                    or (
                        process_pid is not None
                        and (
                            isinstance(process_pid, bool)
                            or not isinstance(process_pid, int)
                            or process_pid <= 0
                        )
                    )
                    or process_group_id is not None
                    or process_group_isolated
                    or process_group_reaped
                )
            )
            or (
                not popen_reached
                and process_group_binding_status
                not in ("not_reached", "popen_attempt_unproven")
            )
            or (
                popen_reached
                and process_group_binding_status == "bound_isolated"
                and (
                    isinstance(process_group_id, bool)
                    or not isinstance(process_group_id, int)
                    or process_group_id <= 0
                    or process_group_id != process_pid
                    or process_group_isolated is not True
                )
            )
            or (
                popen_reached
                and process_group_binding_status == "failed_unproven"
                and (
                    process_group_isolated is not False
                    or process_group_reaped is not False
                    or (
                        process_group_id is not None
                        and (
                            isinstance(process_group_id, bool)
                            or not isinstance(process_group_id, int)
                            or process_group_id <= 0
                            or process_group_id == process_pid
                        )
                    )
                )
            )
            or (
                popen_reached
                and process_group_binding_status
                not in ("bound_isolated", "failed_unproven")
            )
            or (
                stage_returned
                and not (
                    popen_reached
                    and process_reaped
                    and process_group_binding_status == "bound_isolated"
                    and process_group_reaped
                )
            )
            or (lock_released and not lock_acquired)
            or (unreaped_stage is not None and lock_released)
            or (lock_released_before_failure and not (lock_released and stage_returned))
            or (
                popen_reached
                and (
                    not process_reaped
                    or process_group_binding_status != "bound_isolated"
                    or not process_group_reaped
                )
                and lock_released
            )
            or unreaped_stage
            not in (None, "initialize_native_event_core", "train_source63_counterfactual_five_seed")
            or lock_retained_for_unreaped
            is not (
                unreaped_stage is not None and lock_acquired and not lock_released
            )
            or artifacts_frozen is not (unreaped_stage is None)
            or (
                popen_reached
                and (
                    not process_reaped
                    or process_group_binding_status != "bound_isolated"
                    or not process_group_reaped
                )
                and unreaped_stage != "train_source63_counterfactual_five_seed"
            )
            or (
                unreaped_stage == "train_source63_counterfactual_five_seed"
                and not (
                    process_group_binding_status == "popen_attempt_unproven"
                    or (
                        popen_reached
                        and (
                            not process_reaped
                            or process_group_binding_status != "bound_isolated"
                            or not process_group_reaped
                        )
                    )
                )
            )
            or (
                process_group_binding_status == "popen_attempt_unproven"
                and unreaped_stage != "train_source63_counterfactual_five_seed"
            )
            or (unreaped_stage is not None and not lock_acquired)
            or (
                phase == "after_gpu_lock_release_before_terminal_finalize"
                and not lock_released_before_failure
            )
            or (
                phase != "after_gpu_lock_release_before_terminal_finalize"
                and lock_released_before_failure
            )
        ):
            raise RuntimeError("source63 failure phase evidence is inconsistent")
    return receipt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(SHA256_CHARS)
    )


def _contains_sensitive_path_component(path: PurePath) -> bool:
    return any(
        token in component.lower()
        for component in path.parts
        for token in SENSITIVE_PATH_TOKENS
    )


def _reject_sensitive_path_text(value: Any, role: str) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{role} path is missing")
    if _contains_sensitive_path_component(PurePath(value)):
        raise RuntimeError(f"{role} references forbidden Fresh/confirmation input")


def resolve_existing_path(path: Path, *, role: str, directory: bool) -> Path:
    supplied = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_path_component(absolute):
        raise ValueError(f"{role} path is forbidden by the Fresh/confirmation boundary")
    if supplied.is_symlink():
        raise ValueError(f"{role} must be materialized, not a symlink")
    resolved = supplied.resolve(strict=True)
    if _contains_sensitive_path_component(resolved):
        raise ValueError(f"{role} resolves into a forbidden path")
    metadata = resolved.stat()
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{role} must be a {kind}")
    return resolved


def resolve_new_path(path: Path, *, role: str) -> Path:
    supplied = path.expanduser()
    absolute = Path(os.path.abspath(os.fspath(supplied)))
    if _contains_sensitive_path_component(absolute):
        raise ValueError(f"{role} path is forbidden by the Fresh/confirmation boundary")
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"{role} already exists: {absolute}")
    parent = absolute.parent.resolve(strict=True)
    if _contains_sensitive_path_component(parent):
        raise ValueError(f"{role} parent resolves into a forbidden path")
    if not parent.is_dir():
        raise ValueError(f"{role} parent must be a directory")
    return absolute


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(path) from error
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{role} must be a materialized regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{role} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{role} must contain a JSON object")
    return value


def _local_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def implementation_closure(code_root: Path) -> dict[str, dict[str, Any]]:
    scripts = code_root / "scripts"
    queue = [scripts / name for name in ENTRYPOINTS]
    seen: set[Path] = set()
    while queue:
        path = queue.pop().resolve(strict=True)
        if path in seen:
            continue
        if scripts.resolve() not in path.parents or not path.is_file():
            raise RuntimeError(f"implementation escaped code root: {path}")
        seen.add(path)
        for module in _local_imports(path):
            candidate = scripts / (module.replace(".", "/") + ".py")
            if candidate.is_file():
                queue.append(candidate)
    names = {path.name for path in seen}
    if not set(ENTRYPOINTS).issubset(names):
        raise RuntimeError("implementation closure is incomplete")
    return {
        str(path.relative_to(code_root)): {
            "path": str(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(seen)
    }


def python_contract(path: Path) -> dict[str, Any]:
    invocation = path.expanduser()
    if not invocation.exists():
        raise FileNotFoundError(invocation)
    resolved = invocation.resolve(strict=True)
    if _contains_sensitive_path_component(resolved) or not resolved.is_file():
        raise ValueError("python executable is invalid or forbidden")
    if not os.access(resolved, os.X_OK):
        raise PermissionError(resolved)
    return {
        "invocation_path": str(Path(os.path.abspath(os.fspath(invocation)))),
        "resolved_path": str(resolved),
        "resolved_sha256": file_sha256(resolved),
    }


def canonical_python_environment(
    inherited: Mapping[str, str],
    *,
    gpu_index: int,
    omp_threads: int,
) -> dict[str, str]:
    """Return the only environment permitted for watcher/stage interpreters.

    In particular, copying an ambient ``PYTHONPATH`` and then merely setting
    ``PYTHONNOUSERSITE`` is not sufficient: PYTHONPATH is processed separately
    from the user-site directory and can select a different Torch installation.
    """

    if gpu_index < 0 or omp_threads <= 0:
        raise ValueError("canonical environment GPU/thread values are invalid")
    environment = {
        str(key): str(value)
        for key, value in inherited.items()
        if not str(key).upper().startswith("PYTHON")
        and str(key).upper() not in SCRUBBED_ENVIRONMENT_NAMES
    }
    environment.update(FORCED_PYTHON_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["OMP_NUM_THREADS"] = str(omp_threads)
    return environment


def canonical_environment_contract(*, gpu_index: int, omp_threads: int) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "format": ENVIRONMENT_CONTRACT_FORMAT,
        "policy": "remove_all_inherited_python_variables_and_environment_manager_hints_v1",
        "removed_name_prefixes": ["PYTHON"],
        "removed_exact_names": sorted(SCRUBBED_ENVIRONMENT_NAMES),
        "forced_python_environment": dict(sorted(FORCED_PYTHON_ENVIRONMENT.items())),
        "cuda_visible_devices": str(gpu_index),
        "omp_num_threads": str(omp_threads),
        "pythonpath_inherited": False,
        "pythonhome_inherited": False,
        "pythonusersite_enabled": False,
    }
    contract["environment_contract_sha256"] = canonical_sha256(contract)
    return contract


def assert_canonical_python_environment(
    environment: Mapping[str, str], contract: Mapping[str, Any]
) -> None:
    if contract.get("format") != ENVIRONMENT_CONTRACT_FORMAT:
        raise RuntimeError("canonical Python environment contract is invalid")
    expected_hash = contract.get("environment_contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("environment_contract_sha256", None)
    if expected_hash != canonical_sha256(unsigned):
        raise RuntimeError("canonical Python environment contract SHA256 changed")
    expected_python = contract.get("forced_python_environment")
    if not isinstance(expected_python, Mapping):
        raise RuntimeError("canonical Python environment values are missing")
    unexpected = sorted(
        key
        for key in environment
        if key.upper().startswith("PYTHON") and key not in expected_python
    )
    wrong = {
        key: environment.get(key)
        for key, value in expected_python.items()
        if environment.get(key) != value
    }
    leaked = sorted(key for key in SCRUBBED_ENVIRONMENT_NAMES if key in environment)
    if unexpected or wrong or leaked:
        raise RuntimeError(
            "non-canonical Python environment: "
            f"unexpected={unexpected}, wrong={wrong}, leaked={leaked}"
        )
    if environment.get("CUDA_VISIBLE_DEVICES") != contract.get(
        "cuda_visible_devices"
    ) or environment.get("OMP_NUM_THREADS") != contract.get("omp_num_threads"):
        raise RuntimeError("canonical CUDA/OMP environment changed")


def _validate_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(value)
    claimed = runtime.pop("runtime_contract_sha256", None)
    required_strings = (
        "python_executable",
        "python_version",
        "python_prefix",
        "python_base_prefix",
        "torch_version",
    )
    if (
        runtime.get("format") != RUNTIME_CONTRACT_FORMAT
        or runtime.get("isolated") is not True
        or any(not isinstance(runtime.get(key), str) or not runtime[key] for key in required_strings)
        or not all(
            isinstance(runtime.get(name), Mapping)
            and isinstance(runtime[name].get("path"), str)
            and _is_sha256(runtime[name].get("sha256"))
            for name in ("torch_module", "torch_c_module")
        )
        or claimed != canonical_sha256(runtime)
    ):
        raise RuntimeError("isolated Python/Torch runtime contract is invalid")
    return {**runtime, "runtime_contract_sha256": claimed}


def probe_python_runtime(
    python_bin: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Probe Torch with ``-I`` so ambient path/user-site inputs are ignored."""

    completed = subprocess.run(
        [str(python_bin), "-I", "-c", RUNTIME_PROBE_SOURCE],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(environment),
        cwd="/",
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("isolated Python/Torch runtime probe output is invalid")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated Python/Torch runtime probe is not JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError("isolated Python/Torch runtime probe is not an object")
    return _validate_runtime_contract(value)


def current_python_runtime() -> dict[str, Any]:
    """Describe the already-running isolated watcher before adding code paths."""

    import torch

    def module_file(path_value: str) -> dict[str, str]:
        path = Path(path_value).resolve(strict=True)
        return {"path": str(path), "sha256": file_sha256(path)}

    runtime: dict[str, Any] = {
        "format": RUNTIME_CONTRACT_FORMAT,
        "isolated": bool(sys.flags.isolated),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": str(sys.version),
        "python_prefix": str(Path(sys.prefix).resolve(strict=True)),
        "python_base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "torch_module": module_file(str(torch.__file__)),
        "torch_c_module": module_file(str(torch._C.__file__)),
    }
    runtime["runtime_contract_sha256"] = canonical_sha256(runtime)
    return _validate_runtime_contract(runtime)


def assert_runtime_matches(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, role: str
) -> None:
    expected_value = _validate_runtime_contract(expected)
    actual_value = _validate_runtime_contract(actual)
    if actual_value != expected_value:
        raise RuntimeError(
            f"{role} Python/Torch runtime drifted: "
            f"expected torch={expected_value['torch_version']} "
            f"at {expected_value['torch_module']['path']}, "
            f"found torch={actual_value['torch_version']} "
            f"at {actual_value['torch_module']['path']}"
        )


def activate_trusted_scripts_path(code_root: Path) -> Path:
    scripts = (code_root / "scripts").resolve(strict=True)
    if (scripts / "torch.py").exists() or (scripts / "torch").exists():
        raise RuntimeError("trusted scripts path contains a forbidden Torch shadow")
    scripts_text = str(scripts)
    sys.path[:] = [entry for entry in sys.path if entry != scripts_text]
    sys.path.insert(0, scripts_text)
    return scripts


def read_frozen_split(path: Path, *, expected_groups: int = EXPECTED_GROUPS) -> dict[str, Any]:
    value = load_json(path, role="source split")
    if (
        value.get("format") != "etsf_smolvla_schema5_native_source_split_v1"
        or value.get("status") != "frozen_development_split"
        or value.get("task") != EXPECTED_TASK
        or value.get("body") != EXPECTED_BODY
        or value.get("policy") != EXPECTED_POLICY
        or value.get("split_unit") != "requested_seed_logical_group"
        or value.get("fresh_inputs_allowed") is not False
        or value.get("fresh_trajectory_or_label_opened") is not False
    ):
        raise RuntimeError("source split contract is invalid")
    result: dict[str, list[int]] = {}
    for split_name in ("train", "validation", "test"):
        entries = value.get(split_name)
        if not isinstance(entries, list) or any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in entries
        ):
            raise RuntimeError(f"source split {split_name} seeds are invalid")
        result[split_name] = list(entries)
    all_seeds = [*result["train"], *result["validation"], *result["test"]]
    if len(all_seeds) != expected_groups or len(set(all_seeds)) != expected_groups:
        raise RuntimeError("source split must contain exactly 63 unique requested seeds")
    if [len(result[name]) for name in ("train", "validation", "test")] != [44, 14, 5]:
        raise RuntimeError("source63 split must contain exactly 44/14/5 groups")
    return {**value, "all_requested_seeds": all_seeds}


def read_collector_exit_once(exit_path: Path) -> int | None:
    """Read only the terminal sentinel; missing means the collector is running."""

    if not exit_path.exists():
        return None
    if exit_path.is_symlink() or not exit_path.is_file():
        raise RuntimeError("collector run.exit must be a materialized regular file")
    try:
        raw = exit_path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise RuntimeError("collector run.exit is unreadable") from error
    if raw not in ("0", "0\n", "0\r\n"):
        stripped = raw.strip()
        if stripped.isdigit():
            raise RuntimeError(f"collector exited unsuccessfully with code {stripped}")
        raise RuntimeError("collector run.exit must contain exactly one integer exit code")
    return 0


def wait_for_collector_exit(
    exit_path: Path,
    *,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    max_polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait without reading manifest, group names, HDF5, or label data."""

    started = time.monotonic()
    polls = 0
    while True:
        status = read_collector_exit_once(exit_path)
        polls += 1
        state.update(
            {
                "status": "waiting_for_collector_run_exit_zero_no_hdf5_access",
                "collector_exit_polls": polls,
                "last_heartbeat_unix": time.time(),
                "manifest_read": False,
                "hdf5_opened": False,
            }
        )
        atomic_json(state_path, state)
        if status == 0:
            return {
                "exit_code": 0,
                "polls": polls,
                "wait_seconds": time.monotonic() - started,
                "run_exit_path": str(exit_path),
                "run_exit_sha256": file_sha256(exit_path),
                "manifest_read_before_exit_zero": False,
                "hdf5_opened_before_exit_zero": False,
            }
        elapsed = time.monotonic() - started
        if (timeout_seconds > 0 and elapsed >= timeout_seconds) or (
            max_polls is not None and polls >= max_polls
        ):
            raise TimeoutError("timed out waiting for collector run.exit=0")
        sleep(poll_seconds)


def validate_collector_metadata(
    manifest_path: Path,
    *,
    split: Mapping[str, Any],
    event_spec_sha256: str,
    expected_groups: int = EXPECTED_GROUPS,
) -> dict[str, Any]:
    """Authenticate completed JSON metadata without opening any HDF5 file."""

    manifest = load_json(manifest_path, role="completed collector manifest")
    requested = list(split["all_requested_seeds"])
    groups = manifest.get("groups")
    if (
        manifest.get("status") != "complete"
        or manifest.get("schema_version") != EXPECTED_SCHEMA
        or manifest.get("task") != EXPECTED_TASK
        or manifest.get("body") != EXPECTED_BODY
        or manifest.get("policy") != EXPECTED_POLICY
        or manifest.get("requested_seeds") != requested
        or manifest.get("event_spec_sha256") != event_spec_sha256
        or int(manifest.get("completed", -1)) != expected_groups
        or not isinstance(groups, list)
        or len(groups) != expected_groups
        or manifest.get("candidate_count") != EXPECTED_CANDIDATES
        or manifest.get("hidden_dim") != EXPECTED_HIDDEN_DIM
        or manifest.get("action_dim") != EXPECTED_ACTION_DIM
        or manifest.get("action_chunk") != EXPECTED_ACTION_CHUNK
    ):
        raise RuntimeError("collector manifest is incomplete or differs from source63")
    modeling_sha = manifest.get("shared_state_modeling_sha256")
    bridge_sha = manifest.get("shared_state_bridge_sha256")
    if not _is_sha256(modeling_sha) or not _is_sha256(bridge_sha):
        raise RuntimeError("collector manifest lacks modeling/bridge SHA bindings")
    for key in (
        "model_path",
        "checkpoint",
        "vlm_metadata_path",
        "modeling_source",
        "bridge_source",
        "event_spec",
    ):
        _reject_sensitive_path_text(manifest.get(key), f"collector manifest {key}")
    resolved_seeds: list[int] = []
    split_by_seed = {
        int(seed): split_name
        for split_name in ("train", "validation", "test")
        for seed in split[split_name]
    }
    group_records: list[dict[str, Any]] = []
    for index, (row, expected_seed) in enumerate(zip(groups, requested)):
        if not isinstance(row, Mapping):
            raise RuntimeError("collector manifest group row is invalid")
        relative = row.get("path")
        if (
            row.get("index") != index
            or row.get("seed") != expected_seed
            or row.get("status") not in ("collected", "existing")
            or not isinstance(relative, str)
            or Path(relative).name != relative
            or Path(relative).suffix not in (".hdf5", ".h5")
            or _contains_sensitive_path_component(PurePath(relative))
        ):
            raise RuntimeError(f"collector manifest group {index} identity is invalid")
        resolved_seed = row.get("resolved_seed")
        if isinstance(resolved_seed, bool) or not isinstance(resolved_seed, int):
            raise RuntimeError("collector resolved seed is invalid")
        resolved_seeds.append(resolved_seed)
        # Deliberately project only identity/provenance fields.  Completed
        # manifests also contain success/steps; the watcher must never read or
        # branch on those fields, especially for development holdout groups.
        group_records.append(
            {
                "index": index,
                "path": relative,
                "status": row["status"],
                "seed": expected_seed,
                "resolved_seed": resolved_seed,
                "split": split_by_seed[expected_seed],
            }
        )
    if len(set(resolved_seeds)) != expected_groups:
        raise RuntimeError("collector resolved seeds are not unique")
    if manifest.get("resolved_seeds") != resolved_seeds:
        raise RuntimeError("collector resolved seed summary differs from group rows")
    return {
        "manifest_header": {
            "candidate_count": manifest["candidate_count"],
            "hidden_dim": manifest["hidden_dim"],
            "action_chunk": manifest["action_chunk"],
        },
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "requested_seeds": requested,
        "resolved_seeds": resolved_seeds,
        "groups": group_records,
        "modeling_sha256": modeling_sha,
        "bridge_sha256": bridge_sha,
        "event_spec_sha256": event_spec_sha256,
        "hdf5_opened": False,
    }


def audit_group_files(
    collector_root: Path,
    metadata: Mapping[str, Any],
    *,
    group_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Open HDF5 only after the caller has authenticated terminal metadata."""

    manifest_header = metadata["manifest_header"]
    groups_dir = collector_root / "groups"
    if groups_dir.is_symlink() or not groups_dir.is_dir():
        raise RuntimeError("collector groups directory is not materialized")
    expected_names = {str(row["path"]) for row in metadata["groups"]}
    actual_names: set[str] = set()
    for path in groups_dir.iterdir():
        if path.suffix not in (".hdf5", ".h5"):
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"collector HDF5 entry is not materialized: {path}")
        actual_names.add(path.name)
    if actual_names != expected_names:
        raise RuntimeError("collector HDF5 file set differs from completed manifest")
    if group_validator is None:
        # This import is deliberately below the run.exit and completed-manifest
        # gates.  Merely waiting can never import or invoke an HDF5 reader.
        from collect_smolvla_etsf_event_branches import validate_group_file

        group_validator = validate_group_file
    audits: list[dict[str, Any]] = []
    for row in metadata["groups"]:
        path = groups_dir / str(row["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"collector group is missing/materialized incorrectly: {path}")
        before = file_sha256(path)
        if row["split"] == "test":
            # The five development holdout containers remain label sealed.
            # Whole-file hashing reads opaque bytes, not HDF5 datasets.
            verified: dict[str, Any] = {
                "status": "development_holdout_byte_hash_only_hdf5_not_opened",
                "seed": row["seed"],
                "resolved_seed": row["resolved_seed"],
                "label_datasets_opened": 0,
                "labels_used": False,
            }
        else:
            verified = dict(
                group_validator(
                    path,
                    int(row["seed"]),
                    int(manifest_header["candidate_count"]),
                    int(manifest_header["hidden_dim"]),
                    int(manifest_header["action_chunk"]),
                    expected_modeling_sha256=metadata["modeling_sha256"],
                    expected_bridge_sha256=metadata["bridge_sha256"],
                    expected_event_spec_sha256=metadata["event_spec_sha256"],
                )
            )
        after = file_sha256(path)
        if before != after:
            raise RuntimeError(f"collector group changed during self-validation: {path}")
        if verified.get("seed") != row.get("seed") or verified.get(
            "resolved_seed"
        ) != row.get("resolved_seed"):
            raise RuntimeError(f"collector group self-audit differs from manifest: {path}")
        audits.append(
            {
                "index": row["index"],
                "requested_seed": row["seed"],
                "resolved_seed": row["resolved_seed"],
                "split": row["split"],
                "source_path": str(path),
                "relative_path": f"groups/{path.name}",
                "sha256": before,
                "size": path.stat().st_size,
                "self_validation": verified,
            }
        )
    if file_sha256(Path(str(metadata["manifest_path"]))) != metadata["manifest_sha256"]:
        raise RuntimeError("collector manifest changed during HDF5 self-validation")
    return audits


def _copy_file_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    destination.chmod(0o444)


def freeze_source_snapshot(
    output_root: Path,
    *,
    collector_exit: Path,
    manifest_path: Path,
    event_spec: Path,
    source_split: Path,
    metadata: Mapping[str, Any],
    group_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot = output_root / "source_snapshot"
    partial = output_root / ".source_snapshot.partial"
    if snapshot.exists() or snapshot.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError("source snapshot output already exists")
    partial.mkdir(mode=0o700)
    bindings = {
        "run.exit": (collector_exit, file_sha256(collector_exit)),
        "manifest.json": (manifest_path, str(metadata["manifest_sha256"])),
        "event_spec.json": (event_spec, file_sha256(event_spec)),
        "source_split.json": (source_split, file_sha256(source_split)),
    }
    try:
        for relative, (source, expected_sha) in bindings.items():
            if file_sha256(source) != expected_sha:
                raise RuntimeError(f"source changed before snapshot: {source}")
            destination = partial / relative
            _copy_file_new(source, destination)
            if file_sha256(destination) != expected_sha:
                raise RuntimeError(f"snapshot copy hash mismatch: {destination}")
        for audit in group_audits:
            source = Path(str(audit["source_path"]))
            expected_sha = str(audit["sha256"])
            if file_sha256(source) != expected_sha:
                raise RuntimeError(f"source group changed before snapshot: {source}")
            destination = partial / str(audit["relative_path"])
            _copy_file_new(source, destination)
            if file_sha256(destination) != expected_sha:
                raise RuntimeError(f"snapshot group hash mismatch: {destination}")
        (partial / "groups").chmod(0o555)
        partial.chmod(0o555)
        os.rename(partial, snapshot)
    except BaseException:
        raise
    receipt: dict[str, Any] = {
        "format": SNAPSHOT_FORMAT,
        "status": "complete_immutable_source_snapshot",
        "source_manifest_sha256": metadata["manifest_sha256"],
        "event_spec_sha256": bindings["event_spec.json"][1],
        "source_split_sha256": bindings["source_split.json"][1],
        "run_exit_sha256": bindings["run.exit"][1],
        "group_count": len(group_audits),
        "groups": [
            {
                "index": audit["index"],
                "requested_seed": audit["requested_seed"],
                "resolved_seed": audit["resolved_seed"],
                "split": audit["split"],
                "relative_path": audit["relative_path"],
                "sha256": audit["sha256"],
                "size": audit["size"],
            }
            for audit in group_audits
        ],
        "source_files_copied_not_hardlinked": True,
        "source_snapshot_path": str(snapshot),
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "train_validation_hdf_self_validated": sum(
            audit["split"] in ("train", "validation") for audit in group_audits
        ),
        "test_hdf_byte_hashed": sum(
            audit["split"] == "test" for audit in group_audits
        ),
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
        "test_hdf_identity_attrs_opened_by_watcher": 0,
    }
    receipt["snapshot_sha256"] = canonical_sha256(receipt)
    atomic_json(output_root / "source_snapshot_receipt.json", receipt)
    return receipt


def build_stage_commands(
    *,
    python_bin: Path,
    stage_runner: Path,
    launch_plan: Path,
    static_plan_sha256: str,
    runtime_contract_sha256: str,
    initializer: Path,
    trainer: Path,
    output_root: Path,
    snapshot_receipt: Mapping[str, Any],
    modeling_sha256: str,
    bridge_sha256: str,
    num_workers: int,
) -> list[dict[str, Any]]:
    snapshot = Path(str(snapshot_receipt["source_snapshot_path"]))
    initialized = output_root / "smolvla_schema5_native_initialized.pt"
    training = output_root / "counterfactual_training"
    bound_prefix = [
        str(python_bin),
        "-I",
        str(stage_runner),
        "--launch-plan",
        str(launch_plan),
        "--static-plan-sha256",
        static_plan_sha256,
        "--target",
    ]
    initialize_argv = [
        *bound_prefix,
        str(initializer),
        "--",
        "--output",
        str(initialized),
        "--event-spec",
        str(snapshot / "event_spec.json"),
        "--event-spec-sha256",
        str(snapshot_receipt["event_spec_sha256"]),
        "--source-manifest",
        str(snapshot / "manifest.json"),
        "--source-manifest-sha256",
        str(snapshot_receipt["source_manifest_sha256"]),
        "--source-split",
        str(snapshot / "source_split.json"),
        "--source-split-sha256",
        str(snapshot_receipt["source_split_sha256"]),
        "--state-modeling-sha256",
        modeling_sha256,
        "--state-bridge-sha256",
        bridge_sha256,
        "--initialization-seed",
        str(INITIALIZATION_SEED),
    ]
    train_argv = [
        *bound_prefix,
        str(trainer),
        "--",
        "--data",
        str(snapshot),
        "--pretrained",
        str(initialized),
        "--output",
        str(training),
        "--split-manifest",
        str(snapshot / "source_split.json"),
        "--event-spec",
        str(snapshot / "event_spec.json"),
        "--object-names",
        "can",
        "--seeds",
        *[str(seed) for seed in TRAINING_SEEDS],
        "--device",
        "cuda",
        "--amp",
        "bf16",
        "--steps",
        str(TRAINING_STEPS),
        "--early-stopping-patience",
        "0",
        "--unfreeze-semantic",
        "--require-policy-feature-action-bridge",
        "--num-workers",
        str(num_workers),
    ]
    return [
        {
            "stage": "initialize_native_dual_reserved_core",
            "argv": initialize_argv,
            "argv_sha256": canonical_sha256(initialize_argv),
            "uses_gpu": False,
            "output": str(initialized),
            "isolated_python": True,
            "runtime_contract_sha256": runtime_contract_sha256,
        },
        {
            "stage": "train_source63_counterfactual_five_seed",
            "argv": train_argv,
            "argv_sha256": canonical_sha256(train_argv),
            "uses_gpu": True,
            "output": str(training),
            "cuda_amp": "bf16",
            "training_steps": TRAINING_STEPS,
            "training_seeds": list(TRAINING_SEEDS),
            "unfreeze_semantic": True,
            "object_names": ["can"],
            "isolated_python": True,
            "runtime_contract_sha256": runtime_contract_sha256,
        },
    ]


def _gpu_query(gpu_index: int, field: str) -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                f"--query-{field}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"unable to audit GPU {field}") from error
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def gpu_name(gpu_index: int) -> str:
    names = _gpu_query(gpu_index, "gpu=name")
    if len(names) != 1:
        raise RuntimeError(f"unexpected GPU name audit: {names}")
    if "4090" not in names[0]:
        raise RuntimeError(f"source63 training requires RTX 4090, found {names[0]!r}")
    return names[0]


def gpu_uuid(gpu_index: int) -> str:
    values = _gpu_query(gpu_index, "gpu=uuid")
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise RuntimeError(f"unexpected GPU UUID audit: {values}")
    return values[0]


def gpu_audit(gpu_index: int) -> dict[str, Any]:
    """Return one logical GPU observation with stable hardware identity."""

    name = gpu_name(gpu_index)
    uuid = gpu_uuid(gpu_index)
    pids = gpu_compute_pids(gpu_index)
    # Re-read UUID after the process query so a device remap cannot validate an
    # idle sample for a different GPU at the same visible index.
    if gpu_uuid(gpu_index) != uuid:
        raise RuntimeError("GPU UUID changed during idle audit")
    return {"gpu_index": gpu_index, "gpu_name": name, "gpu_uuid": uuid, "compute_pids": pids}


def gpu_compute_pids(gpu_index: int) -> list[int]:
    values = _gpu_query(gpu_index, "compute-apps=pid")
    result: list[int] = []
    for value in values:
        lowered = value.lower()
        if lowered in ("no running processes found", "no running processes found."):
            continue
        if not value.isdigit():
            raise RuntimeError(f"unexpected GPU PID output: {value}")
        result.append(int(value))
    return sorted(set(result))


def _proc_start_ticks(pid: int, raw_stat: bytes) -> int:
    try:
        stat_text = raw_stat.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("external suite parent stat is not ASCII") from error
    close_paren = stat_text.rfind(")")
    if close_paren < 0 or not stat_text.startswith(f"{pid} ("):
        raise RuntimeError("external suite parent stat record is malformed")
    tail_fields = stat_text[close_paren + 2 :].split()
    if len(tail_fields) <= 19 or not tail_fields[19].isdigit():
        raise RuntimeError("external suite parent start ticks are missing")
    start_ticks = int(tail_fields[19])
    if start_ticks <= 0:
        raise RuntimeError("external suite parent start ticks are invalid")
    return start_ticks


def read_external_process_identity(
    pid: int, *, proc_root: Path = Path("/proc")
) -> dict[str, Any] | None:
    """Read a stable Linux process identity without trusting PID alone.

    ``/proc/<pid>/stat`` is read before and after ``cmdline``.  Only PID,
    starttime and kernel boot ID define identity; mutable accounting fields in
    stat are deliberately ignored.  A changed starttime indicates PID reuse.
    """

    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("external suite parent PID must be a positive integer")
    process_root = proc_root / str(pid)
    stat_path = process_root / "stat"
    cmdline_path = process_root / "cmdline"
    boot_id_path = proc_root / "sys" / "kernel" / "random" / "boot_id"
    try:
        boot_id_before = boot_id_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        raise RuntimeError("kernel boot ID is unavailable during parent audit")
    except OSError as error:
        raise RuntimeError("unable to audit kernel boot ID") from error
    try:
        first_stat = stat_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"unable to audit external suite parent PID {pid}") from error
    try:
        cmdline = cmdline_path.read_bytes()
        second_stat = stat_path.read_bytes()
    except FileNotFoundError as error:
        # A partial /proc read is absence only if the stable identity record is
        # also gone.  Missing cmdline while stat still exists is not evidence
        # that the protected suite exited.
        try:
            stat_path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as stat_error:
            raise RuntimeError(
                f"unable to confirm external suite parent PID {pid} absence"
            ) from stat_error
        raise RuntimeError(
            "external suite parent audit became partial while PID still exists"
        ) from error
    except OSError as error:
        raise RuntimeError(f"unable to audit external suite parent PID {pid}") from error
    try:
        boot_id_after = boot_id_path.read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("kernel boot ID disappeared during parent audit") from error
    if boot_id_before != boot_id_after:
        raise RuntimeError("kernel boot ID changed during parent audit")
    first_start_ticks = _proc_start_ticks(pid, first_stat)
    second_start_ticks = _proc_start_ticks(pid, second_stat)
    if first_start_ticks != second_start_ticks:
        raise RuntimeError("external suite parent PID was reused during /proc audit")
    if not cmdline or not cmdline.endswith(b"\0") or len(boot_id_before) != 36:
        raise RuntimeError("external suite parent identity is incomplete")
    cmdline_sha256 = hashlib.sha256(cmdline).hexdigest()
    cmdline_arguments = [
        argument.decode("utf-8", errors="surrogateescape")
        for argument in cmdline.rstrip(b"\0").split(b"\0")
    ]
    return {
        "pid": pid,
        "start_ticks": first_start_ticks,
        "boot_id": boot_id_before,
        "cmdline_sha256": cmdline_sha256,
        "cmdline_arguments": cmdline_arguments,
    }


def external_suite_parent_guard(
    args: argparse.Namespace, *, require_live: bool = True
) -> dict[str, Any]:
    """Build and authenticate the optional immutable parent-process guard."""

    pid = getattr(args, "external_suite_parent_pid", None)
    start_ticks = getattr(args, "external_suite_parent_start_ticks", None)
    boot_id = getattr(args, "external_suite_parent_boot_id", None)
    cmdline_sha256 = getattr(args, "external_suite_parent_cmdline_sha256", None)
    script = getattr(args, "external_suite_parent_script", None)
    script_sha256 = getattr(args, "external_suite_parent_script_sha256", None)
    values = (pid, start_ticks, boot_id, cmdline_sha256, script, script_sha256)
    if all(value is None for value in values):
        return {
            "format": EXTERNAL_SUITE_GUARD_FORMAT,
            "enabled": False,
            "idle_confirmations_required_after_exit": 2,
        }
    if any(value is None for value in values):
        raise ValueError(
            "external suite parent guard requires PID, start ticks, boot ID, cmdline "
            "SHA256, exact script path, and script SHA256 together"
        )
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
        or not isinstance(boot_id, str)
        or len(boot_id) != 36
        or not _is_sha256(cmdline_sha256)
        or not _is_sha256(script_sha256)
    ):
        raise ValueError("external suite parent guard identity is invalid")
    script_path = resolve_existing_path(
        Path(script), role="external suite parent script", directory=False
    )
    if not Path(script).is_absolute():
        raise ValueError("external suite parent script path must be absolute")
    if file_sha256(script_path) != script_sha256:
        raise RuntimeError("external suite parent script SHA256 changed")
    contract: dict[str, Any] = {
        "format": EXTERNAL_SUITE_GUARD_FORMAT,
        "enabled": True,
        "pid": pid,
        "start_ticks": start_ticks,
        "boot_id": boot_id,
        "cmdline_sha256": cmdline_sha256,
        "script_path": str(script_path),
        "script_sha256": script_sha256,
        "identity_policy": "linux_boot_id_proc_starttime_raw_cmdline_and_script_sha256_v2",
        "pid_reuse_policy": "fail_closed",
        "idle_confirmations_required_after_exit": 2,
    }
    identity = read_external_process_identity(pid)
    if identity is None and require_live:
        raise RuntimeError("external suite parent is absent during initial preflight")
    if identity is not None:
        verify_external_suite_parent_identity(contract, identity)
    contract["guard_contract_sha256"] = canonical_sha256(contract)
    return contract


def verify_external_suite_parent_identity(
    contract: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    """Reject a live process unless it is exactly the frozen suite parent."""

    if contract.get("format") != EXTERNAL_SUITE_GUARD_FORMAT or not contract.get(
        "enabled"
    ):
        raise RuntimeError("external suite parent guard contract is invalid")
    arguments = identity.get("cmdline_arguments")
    script_path = contract.get("script_path")
    if (
        identity.get("pid") != contract.get("pid")
        or identity.get("start_ticks") != contract.get("start_ticks")
        or identity.get("boot_id") != contract.get("boot_id")
        or identity.get("cmdline_sha256") != contract.get("cmdline_sha256")
        or not isinstance(arguments, list)
        or not isinstance(script_path, str)
        or arguments.count(script_path) != 1
    ):
        raise RuntimeError(
            "external suite parent PID was reused or its frozen identity changed"
        )


def external_suite_parent_alive(
    contract: Mapping[str, Any],
    *,
    identity_reader: Callable[[int], Mapping[str, Any] | None] = read_external_process_identity,
) -> bool:
    if contract.get("format") != EXTERNAL_SUITE_GUARD_FORMAT:
        raise RuntimeError("external suite parent guard format changed")
    if contract.get("enabled") is not True:
        return False
    claimed_sha = contract.get("guard_contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("guard_contract_sha256", None)
    if claimed_sha != canonical_sha256(unsigned):
        raise RuntimeError("external suite parent guard contract SHA256 changed")
    pid = contract.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RuntimeError("external suite parent guard PID changed")
    identity = identity_reader(pid)
    if identity is None:
        return False
    verify_external_suite_parent_identity(contract, identity)
    return True


def wait_for_idle_4090(
    *,
    gpu_index: int,
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    external_parent_guard: Mapping[str, Any] | None = None,
    expected_gpu_identity: Mapping[str, Any] | None = None,
    identity_reader: Callable[[int], Mapping[str, Any] | None] = read_external_process_identity,
    gpu_audit_reader: Callable[[int], Mapping[str, Any]] = gpu_audit,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall_time: Callable[[], float] = time.time,
) -> dict[str, Any]:
    expected_gpu = dict(expected_gpu_identity or {})
    name = expected_gpu.get("gpu_name")
    uuid = expected_gpu.get("gpu_uuid")
    if not isinstance(name, str) or "4090" not in name or not isinstance(uuid, str):
        raise RuntimeError("frozen RTX4090 identity is missing")
    started = monotonic()
    checks = 0
    idle_confirmations = 0
    parent_gone = False
    valid_idle_observations: list[dict[str, Any]] = []
    guard = dict(
        external_parent_guard
        or {
            "format": EXTERNAL_SUITE_GUARD_FORMAT,
            "enabled": False,
            "idle_confirmations_required_after_exit": 2,
        }
    )
    required_idle = guard.get("idle_confirmations_required_after_exit")
    if required_idle != 2:
        raise RuntimeError("RTX4090 idle confirmation contract changed")
    observation_chain = canonical_sha256(
        {"guard": guard, "gpu_index": gpu_index, "gpu_name": name, "gpu_uuid": uuid}
    )
    initial_audit = {
        "format": GPU_IDLE_AUDIT_FORMAT,
        "gpu_index": gpu_index,
        "gpu_name": name,
        "gpu_uuid": uuid,
        "compute_pids": None,
        "checks": 0,
        "wait_seconds": 0.0,
        "external_suite_parent_guard_enabled": guard.get("enabled") is True,
        "external_suite_parent_pid": guard.get("pid"),
        "external_suite_parent_alive": None,
        "external_suite_parent_alive_before_gpu_query": None,
        "external_suite_parent_alive_after_gpu_query": None,
        "external_suite_parent_gone_latched": False,
        "idle_confirmations": 0,
        "idle_confirmations_required": required_idle,
        "observation_chain_sha256": observation_chain,
        "valid_idle_observations": [],
        "status": "waiting_for_guarded_rtx4090_release",
    }
    initial_audit["gpu_idle_release_audit_sha256"] = canonical_sha256(initial_audit)
    state["gpu_idle_audit"] = initial_audit
    state["status"] = "waiting_for_guarded_rtx4090_release_first_observation"
    state["last_heartbeat_unix"] = wall_time()
    atomic_json(state_path, state)
    while True:
        parent_alive_before = external_suite_parent_alive(
            guard, identity_reader=identity_reader
        )
        if parent_gone and parent_alive_before:
            raise RuntimeError("external suite parent PID reappeared after confirmed exit")
        gpu_observation = dict(gpu_audit_reader(gpu_index))
        if (
            gpu_observation.get("gpu_index") != gpu_index
            or gpu_observation.get("gpu_name") != name
            or gpu_observation.get("gpu_uuid") != uuid
        ):
            raise RuntimeError("RTX4090 identity changed during idle wait")
        pids = gpu_observation.get("compute_pids")
        if not isinstance(pids, list) or any(
            isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in pids
        ):
            raise RuntimeError("RTX4090 compute PID audit is invalid")
        parent_alive_after = external_suite_parent_alive(
            guard, identity_reader=identity_reader
        )
        if (parent_gone or not parent_alive_before) and parent_alive_after:
            raise RuntimeError("external suite parent PID reappeared during GPU idle audit")
        parent_alive = parent_alive_before or parent_alive_after
        if not parent_alive_after:
            parent_gone = True
        checks += 1
        if parent_alive or pids:
            idle_confirmations = 0
            valid_idle_observations = []
        else:
            idle_confirmations += 1
        observation = {
            "check": checks,
            "parent_alive_before_gpu_query": parent_alive_before,
            "compute_pids": pids,
            "parent_alive_after_gpu_query": parent_alive_after,
        }
        observation_chain = canonical_sha256(
            {"previous": observation_chain, "observation": observation}
        )
        if not parent_alive and not pids:
            valid_idle_observations.append(
                {
                    **observation,
                    "gpu_index": gpu_index,
                    "gpu_name": name,
                    "gpu_uuid": uuid,
                    "observed_unix": wall_time(),
                    "observation_chain_sha256": observation_chain,
                }
            )
            valid_idle_observations = valid_idle_observations[-required_idle:]
        audit = {
            "format": GPU_IDLE_AUDIT_FORMAT,
            "gpu_index": gpu_index,
            "gpu_name": name,
            "gpu_uuid": uuid,
            "compute_pids": pids,
            "checks": checks,
            "wait_seconds": monotonic() - started,
            "external_suite_parent_guard_enabled": guard.get("enabled") is True,
            "external_suite_parent_pid": guard.get("pid"),
            "external_suite_parent_alive": parent_alive,
            "external_suite_parent_alive_before_gpu_query": parent_alive_before,
            "external_suite_parent_alive_after_gpu_query": parent_alive_after,
            "external_suite_parent_gone_latched": parent_gone,
            "idle_confirmations": idle_confirmations,
            "idle_confirmations_required": required_idle,
            "observation_chain_sha256": observation_chain,
            "valid_idle_observations": valid_idle_observations,
        }
        release_ready = (
            not parent_alive
            and not pids
            and idle_confirmations >= required_idle
            and len(valid_idle_observations) == required_idle
        )
        audit["status"] = (
            "complete_two_idle_samples_released_for_training"
            if release_ready
            else "waiting_for_guarded_rtx4090_release"
        )
        audit["gpu_idle_release_audit_sha256"] = canonical_sha256(audit)
        state["gpu_idle_audit"] = audit
        state["status"] = (
            "waiting_for_external_suite_parent_exit_before_rtx4090"
            if parent_alive
            else "waiting_for_two_consecutive_exclusive_idle_rtx4090_audits"
        )
        state["last_heartbeat_unix"] = wall_time()
        atomic_json(state_path, state)
        if release_ready:
            return audit
        if timeout_seconds > 0 and monotonic() - started >= timeout_seconds:
            raise TimeoutError("timed out waiting for an idle RTX 4090")
        sleep(poll_seconds)


def audit_training_start_guard(
    plan: Mapping[str, Any],
    idle_audit: Mapping[str, Any],
    *,
    identity_reader: Callable[[int], Mapping[str, Any] | None] = read_external_process_identity,
    gpu_audit_reader: Callable[[int], Mapping[str, Any]] = gpu_audit,
    wall_time: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Revalidate the dynamic release immediately before training Popen."""

    idle = dict(idle_audit)
    claimed_idle_sha = idle.pop("gpu_idle_release_audit_sha256", None)
    if (
        idle.get("status") != "complete_two_idle_samples_released_for_training"
        or not _is_sha256(claimed_idle_sha)
        or claimed_idle_sha != canonical_sha256(idle)
        or idle.get("idle_confirmations") != 2
        or len(idle.get("valid_idle_observations", [])) != 2
    ):
        raise RuntimeError("GPU idle release audit is incomplete or changed")
    guard = plan.get("external_suite_parent_guard")
    if not isinstance(guard, Mapping):
        raise RuntimeError("external suite parent guard is missing before training")
    script = Path(str(guard.get("script_path", "")))
    if (
        guard.get("enabled") is True
        and (script.is_symlink() or not script.is_file() or file_sha256(script) != guard.get("script_sha256"))
    ):
        raise RuntimeError("external suite parent script changed before training Popen")
    if external_suite_parent_alive(guard, identity_reader=identity_reader):
        raise RuntimeError("external suite parent reappeared before training Popen")
    gpu_index = int(plan.get("gpu_index", -1))
    observed_gpu = dict(gpu_audit_reader(gpu_index))
    expected_gpu = plan.get("gpu_identity")
    if (
        not isinstance(expected_gpu, Mapping)
        or any(
            observed_gpu.get(key) != expected_gpu.get(key)
            for key in ("gpu_index", "gpu_name", "gpu_uuid")
        )
        or observed_gpu.get("compute_pids") != []
    ):
        raise RuntimeError("RTX4090 was not exclusively idle before training Popen")
    if external_suite_parent_alive(guard, identity_reader=identity_reader):
        raise RuntimeError("external suite parent reappeared during training start audit")
    result: dict[str, Any] = {
        "format": "etsf_guarded_rtx4090_training_start_audit_v1",
        "status": "complete_parent_absent_gpu_idle_script_unchanged",
        "gpu_idle_release_audit_sha256": claimed_idle_sha,
        "external_suite_parent_guard_contract_sha256": guard.get(
            "guard_contract_sha256"
        ),
        "gpu_identity": {
            key: observed_gpu[key] for key in ("gpu_index", "gpu_name", "gpu_uuid")
        },
        "compute_pids": [],
        "script_sha256": guard.get("script_sha256"),
        "observed_unix": wall_time(),
    }
    result["training_start_guard_audit_sha256"] = canonical_sha256(result)
    return result


def acquire_lock(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"concurrent/stale launcher lock exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def release_owned_lock(
    path: Path, token: str, *, strict: bool = False
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "format": GPU_LOCK_RELEASE_AUDIT_FORMAT,
        "path": str(path),
        "pid": os.getpid(),
        "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "released": False,
        "observed_unix": time.time(),
    }
    try:
        value = load_json(path, role="launcher GPU lock")
        audit["observed_owner_pid"] = value.get("pid")
        audit["observed_token_sha256"] = (
            hashlib.sha256(str(value.get("token", "")).encode("utf-8")).hexdigest()
        )
        if value.get("token") != token or value.get("pid") != os.getpid():
            raise RuntimeError("GPU lock ownership changed before release")
        path.unlink()
        if path.exists() or path.is_symlink():
            raise RuntimeError("GPU lock still exists after owned release")
        audit["status"] = "released_exact_owned_gpu_lock"
        audit["released"] = True
        audit["released_unix"] = time.time()
    except (OSError, RuntimeError) as error:
        audit["status"] = "release_failed_closed"
        audit["error_type"] = type(error).__name__
        audit["error"] = str(error)
        if strict:
            raise
    audit["release_audit_sha256"] = canonical_sha256(audit)
    return audit


def validate_initialized_output(path: Path) -> dict[str, Any]:
    from initialize_smolvla_schema5_native_event_core import verify_initialized_core

    audit = verify_initialized_core(path)
    if audit.get("status") != "initialized_data_blind_untrained_not_transfer_ready":
        raise RuntimeError("native initializer output status is invalid")
    return {**audit, "file_sha256": file_sha256(path)}


def validate_training_output(
    path: Path, *, expected_pretrained_sha256: str
) -> dict[str, Any]:
    from etsf_torch_weights_only_compat_v1 import load_numpy_weights_only
    from openvla_etsf_event_world_model import EventWorldModelConfig
    from train_openvla_etsf_counterfactual import (
        validate_reserved_rows_source_only_proof,
        validate_reserved_target_rows,
    )
    from etsf_policy_feature_action_bridge import (
        validate_checkpoint_policy_bridge_header,
    )

    manifest_path = path / "ensemble_manifest.json"
    manifest = load_json(manifest_path, role="counterfactual ensemble manifest")
    members = manifest.get("members")
    if (
        manifest.get("format") != "etsf_counterfactual_ensemble_v1"
        or not isinstance(members, list)
        or len(members) != len(TRAINING_SEEDS)
        or [row.get("seed") for row in members] != list(TRAINING_SEEDS)
        or manifest.get("test_policy")
        != "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
    ):
        raise RuntimeError("counterfactual ensemble output is incomplete")
    proofs: list[dict[str, Any]] = []
    bridge_contract_shas: list[str] = []
    training_log_sha256: list[str] = []
    for member, seed in zip(members, TRAINING_SEEDS):
        if not isinstance(member, Mapping):
            raise RuntimeError("counterfactual member record is invalid")
        member_path = Path(str(member.get("path", ""))).resolve(strict=True)
        if path.resolve() not in member_path.parents:
            raise RuntimeError("counterfactual member escaped training output")
        if file_sha256(member_path) != member.get("sha256"):
            raise RuntimeError("counterfactual member SHA changed")
        checkpoint = load_numpy_weights_only(member_path)
        if not isinstance(checkpoint, Mapping) or checkpoint.get("seed") != seed:
            raise RuntimeError("counterfactual member checkpoint is invalid")
        config = EventWorldModelConfig.from_dict(checkpoint["config"])
        bridge = validate_checkpoint_policy_bridge_header(
            checkpoint["config"], checkpoint.get("contract", {})
        )
        if bridge.get("policy") != "smolvla" or bridge.get("policy_row") != 0:
            raise RuntimeError("counterfactual member policy bridge is invalid")
        bridge_contract_shas.append(str(bridge["contract_sha256"]))
        rows = validate_reserved_target_rows(checkpoint, config)
        proof = validate_reserved_rows_source_only_proof(checkpoint, rows)
        if (
            proof is None
            or proof.get("input_pretrained_checkpoint_sha256")
            != expected_pretrained_sha256
            or proof.get("source_training_steps") <= 0
            or proof.get("source_training_steps") > TRAINING_STEPS
            or proof.get("source_training_groups") != 44
            or proof.get("target_data_read") is not False
            or proof.get("target_labels_read") is not False
        ):
            raise RuntimeError("counterfactual dual reserved-row proof is invalid")
        train_log = member_path.parent / "train_log.jsonl"
        if train_log.is_symlink() or not train_log.is_file():
            raise RuntimeError("counterfactual member training log is missing")
        line_count = 0
        with train_log.open("r", encoding="utf-8") as handle:
            for line_count, line in enumerate(handle, start=1):
                try:
                    log_row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError("counterfactual training log is malformed") from error
                if not isinstance(log_row, Mapping) or log_row.get("step") != line_count:
                    raise RuntimeError("counterfactual training log step sequence changed")
        if line_count != TRAINING_STEPS:
            raise RuntimeError("counterfactual member did not execute exactly 3000 steps")
        training_log_sha256.append(file_sha256(train_log))
        proofs.append(proof)
    ensemble_record = manifest.get("ensemble_checkpoint")
    if not isinstance(ensemble_record, Mapping):
        raise RuntimeError("ensemble checkpoint record is missing")
    ensemble_path = Path(str(ensemble_record.get("path", ""))).resolve(strict=True)
    if path.resolve() not in ensemble_path.parents or file_sha256(
        ensemble_path
    ) != ensemble_record.get("sha256"):
        raise RuntimeError("ensemble checkpoint provenance changed")
    ensemble_checkpoint = load_numpy_weights_only(ensemble_path)
    ensemble_bridge = validate_checkpoint_policy_bridge_header(
        ensemble_checkpoint.get("config", {}),
        ensemble_checkpoint.get("contract", {}),
    )
    if (
        ensemble_bridge.get("policy") != "smolvla"
        or len(set(bridge_contract_shas)) != 1
        or ensemble_bridge.get("contract_sha256") != bridge_contract_shas[0]
    ):
        raise RuntimeError("counterfactual ensemble policy bridge is inconsistent")
    return {
        "status": "complete_verified_source63_counterfactual_training",
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "ensemble_checkpoint": str(ensemble_path),
        "ensemble_checkpoint_sha256": ensemble_record["sha256"],
        "member_count": len(members),
        "member_seeds": list(TRAINING_SEEDS),
        "member_proof_sha256": [proof["proof_sha256"] for proof in proofs],
        "policy_feature_action_bridge_sha256": ensemble_bridge[
            "contract_sha256"
        ],
        "member_training_log_sha256": training_log_sha256,
        "member_training_steps_verified": [TRAINING_STEPS] * len(proofs),
        "target_data_read": False,
        "target_labels_read": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
        "test_hdf_identity_attrs_opened": 5,
    }


def _process_group_exists(process_group_id: int) -> bool:
    if isinstance(process_group_id, bool) or process_group_id <= 0:
        raise ValueError("process group id must be positive")
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_gone(process_group_id: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while _process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _terminate_process_group(process_group_id: int) -> bool:
    if not _process_group_exists(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    if _wait_process_group_gone(process_group_id, 10.0):
        return True
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_process_group_gone(process_group_id, 10.0)


def run_subprocess_stage(
    stage: Mapping[str, Any],
    *,
    output_root: Path,
    environment: Mapping[str, str],
    state: dict[str, Any],
    state_path: Path,
    poll_seconds: float,
    timeout_seconds: float,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    pre_popen_guard: Callable[[], Mapping[str, Any]] | None = None,
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(stage["stage"])
    log_path = output_root / "logs" / f"{name}.log"
    receipt_path = output_root / "stage_receipts" / f"{name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError(f"stage output already exists: {name}")
    started_unix = time.time()
    started_monotonic = time.monotonic()
    stage_error: BaseException | None = None
    return_code: int | None = None
    runtime_before: dict[str, Any] | None = None
    runtime_after: dict[str, Any] | None = None
    pre_popen_guard_audit: dict[str, Any] | None = None
    popen_unix: float | None = None
    process: subprocess.Popen[Any] | None = None
    process_group_id: int | None = None
    process_group_isolated = False
    running: dict[str, Any] | None = None
    cleanup_error: BaseException | None = None
    if lifecycle is None:
        lifecycle = {}
    lifecycle.update(
        {
            "pre_popen_guard_started": False,
            "pre_popen_guard_completed": False,
            "popen_attempted": False,
            "popen_reached": False,
            "process_pid": None,
            "process_reaped": False,
            "process_group_id": None,
            "process_group_isolated": False,
            "process_group_reaped": False,
            "returncode": None,
        }
    )
    with log_path.open("x", encoding="utf-8") as log_handle:
        try:
            if expected_runtime_contract is not None:
                runtime_before = probe_python_runtime(
                    Path(str(stage["argv"][0])), environment
                )
                assert_runtime_matches(
                    expected_runtime_contract,
                    runtime_before,
                    role=f"{name} pre-start probe",
                )
            if pre_popen_guard is not None:
                if lifecycle is not None:
                    lifecycle["pre_popen_guard_started"] = True
                guarded = pre_popen_guard()
                if not isinstance(guarded, Mapping):
                    raise RuntimeError("pre-Popen guard did not return an audit mapping")
                pre_popen_guard_audit = dict(guarded)
                if lifecycle is not None:
                    lifecycle["pre_popen_guard_completed"] = True
            popen_unix = time.time()
            if lifecycle is not None:
                lifecycle["popen_attempted"] = True
            process = subprocess.Popen(
                list(stage["argv"]),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=dict(environment),
                close_fds=True,
                start_new_session=True,
            )
            lifecycle.update(
                {
                    "popen_reached": True,
                    "process_pid": process.pid,
                    "popen_unix": popen_unix,
                }
            )
            process_group_id = os.getpgid(process.pid)
            process_group_isolated = process_group_id == process.pid
            started_monotonic = time.monotonic()
            lifecycle.update(
                {
                    "process_group_id": process_group_id,
                    "process_group_isolated": process_group_isolated,
                }
            )
            if not process_group_isolated:
                raise RuntimeError("stage process did not enter its own process group")
            running = {
                "format": FORMAT,
                "stage": name,
                "status": "running",
                "pid": process.pid,
                "started_unix": started_unix,
                "popen_unix": popen_unix,
                "argv": list(stage["argv"]),
                "argv_sha256": stage["argv_sha256"],
                "log": str(log_path),
            }
            if pre_popen_guard_audit is not None:
                running["pre_popen_guard_audit"] = pre_popen_guard_audit
                running["training_start_guard_audit_sha256"] = (
                    pre_popen_guard_audit.get("training_start_guard_audit_sha256")
                )
            for security_field in (
                "external_suite_parent_guard_contract_sha256",
                "gpu_idle_release_audit_sha256",
                "training_start_guard_audit_sha256",
            ):
                if security_field in stage:
                    running[security_field] = stage[security_field]
            atomic_json(receipt_path, running)
            state.update(
                {
                    "status": f"running_{name}",
                    "current_stage": name,
                    "stage_pid": process.pid,
                    "training_start_guard_audit": pre_popen_guard_audit,
                    "last_heartbeat_unix": time.time(),
                }
            )
            atomic_json(state_path, state)
            while process.poll() is None:
                if (
                    timeout_seconds > 0
                    and time.monotonic() - started_monotonic >= timeout_seconds
                ):
                    raise TimeoutError(f"stage timed out: {name}")
                state["last_heartbeat_unix"] = time.time()
                state["stage_elapsed_seconds"] = time.monotonic() - started_monotonic
                state["stage_log_bytes"] = log_path.stat().st_size
                atomic_json(state_path, state)
                time.sleep(poll_seconds)
            return_code = process.returncode
            if expected_runtime_contract is not None:
                runtime_after = probe_python_runtime(
                    Path(str(stage["argv"][0])), environment
                )
                assert_runtime_matches(
                    expected_runtime_contract,
                    runtime_after,
                    role=f"{name} post-exit probe",
                )
        except BaseException as error:
            stage_error = error
        finally:
            if process is not None:
                try:
                    if stage_error is not None and process.poll() is None:
                        if process_group_isolated and process_group_id is not None:
                            try:
                                os.killpg(process_group_id, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        else:
                            process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if process_group_isolated and process_group_id is not None:
                            try:
                                os.killpg(process_group_id, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            process.kill()
                        process.wait(timeout=10)
                    return_code = process.returncode
                except BaseException as error:
                    cleanup_error = error
                reaped = isinstance(process.returncode, int)
                group_reaped = False
                if process_group_isolated and process_group_id is not None:
                    try:
                        descendants_remained = _process_group_exists(process_group_id)
                        if descendants_remained:
                            if stage_error is None:
                                stage_error = RuntimeError(
                                    f"stage left live descendant processes: {name}"
                                )
                            if not _terminate_process_group(process_group_id):
                                raise RuntimeError(
                                    f"stage process group could not be reaped: {name}"
                                )
                        group_reaped = not _process_group_exists(process_group_id)
                    except BaseException as error:
                        if cleanup_error is None:
                            cleanup_error = error
                if lifecycle is not None:
                    lifecycle.update(
                        {
                            "process_reaped": reaped,
                            "process_group_reaped": group_reaped,
                            "returncode": process.returncode,
                        }
                    )
                if (not reaped or not group_reaped) and cleanup_error is None:
                    cleanup_error = RuntimeError(
                        f"stage process tree could not be proven reaped: {name}"
                    )
                if cleanup_error is not None and stage_error is None:
                    stage_error = cleanup_error
    if running is None:
        running = {
            "format": FORMAT,
            "stage": name,
            "status": "failed_closed_before_start",
            "pid": process.pid if process is not None else None,
            "started_unix": started_unix,
            "popen_unix": popen_unix,
            "argv": list(stage["argv"]),
            "argv_sha256": stage["argv_sha256"],
            "log": str(log_path),
        }
        if pre_popen_guard_audit is not None:
            running["pre_popen_guard_audit"] = pre_popen_guard_audit
            running["training_start_guard_audit_sha256"] = (
                pre_popen_guard_audit.get("training_start_guard_audit_sha256")
            )
        for security_field in (
            "external_suite_parent_guard_contract_sha256",
            "gpu_idle_release_audit_sha256",
            "training_start_guard_audit_sha256",
        ):
            if security_field in stage:
                running[security_field] = stage[security_field]
    log_path.chmod(0o444)
    result = {
        **running,
        "status": "complete" if return_code == 0 and stage_error is None else "failed_closed",
        "returncode": return_code,
        "process_reaped": (
            lifecycle.get("process_reaped") is True if lifecycle is not None else None
        ),
        "process_group_id": process_group_id,
        "process_group_isolated": process_group_isolated,
        "process_group_reaped": (
            lifecycle.get("process_group_reaped") is True
            if lifecycle is not None
            else None
        ),
        "finished_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "log_sha256": file_sha256(log_path),
        "log_bytes": log_path.stat().st_size,
        "runtime_contract_sha256_before": (
            runtime_before.get("runtime_contract_sha256")
            if runtime_before is not None
            else None
        ),
        "runtime_contract_sha256_after": (
            runtime_after.get("runtime_contract_sha256")
            if runtime_after is not None
            else None
        ),
    }
    if stage_error is not None:
        result["error_type"] = type(stage_error).__name__
        result["error"] = str(stage_error)
    if cleanup_error is not None:
        result["cleanup_error_type"] = type(cleanup_error).__name__
        result["cleanup_error"] = str(cleanup_error)
    atomic_json(receipt_path, result)
    if stage_error is not None:
        raise stage_error
    if return_code != 0:
        raise RuntimeError(f"stage {name} failed with exit {return_code}; see {log_path}")
    return result


def recursive_artifact_inventory(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"output artifact symlink is forbidden: {path}")
        if not path.is_file() or path.name in (
            "launch_state.json",
            "final_receipt.json",
            "failure_receipt.json",
            "artifact_inventory.json",
        ):
            continue
        records.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    result: dict[str, Any] = {
        "format": "etsf_smolvla_schema5_source63_artifact_inventory_v1",
        "status": "complete_pre_freeze_inventory",
        "file_count": len(records),
        "files": records,
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def tree_freeze_contract(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    directories: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"cannot freeze symlink artifact: {path}")
        relative = str(path.relative_to(root))
        if path.is_file():
            if path.name in TERMINAL_RECEIPT_NAMES and path.parent == root:
                continue
            files.append(
                {
                    "relative_path": relative,
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        elif path.is_dir():
            directories.append(relative)
    contract: dict[str, Any] = {
        "format": TREE_FREEZE_CONTRACT_FORMAT,
        "output_root": str(root.resolve(strict=True)),
        "files": files,
        "directories": directories,
        "excluded_terminal_receipt_names": sorted(TERMINAL_RECEIPT_NAMES),
        "target_file_mode": 0o444,
        "target_directory_mode": 0o555,
        "terminal_prepublication_mode": 0,
        "terminal_published_mode": 0o444,
    }
    contract["tree_freeze_contract_sha256"] = canonical_sha256(contract)
    return contract


def _validate_tree_freeze_contract_semantics(value: Mapping[str, Any]) -> None:
    _verify_embedded_canonical_sha(
        value,
        "tree_freeze_contract_sha256",
        role="terminal tree freeze contract",
    )
    files = value.get("files")
    directories = value.get("directories")
    if (
        value.get("format") != TREE_FREEZE_CONTRACT_FORMAT
        or not isinstance(value.get("output_root"), str)
        or not Path(value["output_root"]).is_absolute()
        or value.get("excluded_terminal_receipt_names")
        != sorted(TERMINAL_RECEIPT_NAMES)
        or value.get("target_file_mode") != 0o444
        or value.get("target_directory_mode") != 0o555
        or value.get("terminal_prepublication_mode") != 0
        or value.get("terminal_published_mode") != 0o444
        or not isinstance(files, list)
        or not files
        or not isinstance(directories, list)
        or directories != sorted(set(directories))
    ):
        raise RuntimeError("terminal tree freeze contract semantics are incomplete")
    relative_paths: list[str] = []
    for record in files:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("relative_path"), str)
            or not record.get("relative_path")
            or PurePath(record["relative_path"]).is_absolute()
            or PurePath(record["relative_path"]).name in TERMINAL_RECEIPT_NAMES
            or isinstance(record.get("size"), bool)
            or not isinstance(record.get("size"), int)
            or record.get("size") < 0
            or not _is_sha256(record.get("sha256"))
        ):
            raise RuntimeError("terminal tree freeze file record is invalid")
        relative_paths.append(record["relative_path"])
    if relative_paths != sorted(set(relative_paths)):
        raise RuntimeError("terminal tree freeze file records are not canonical")
    for relative in directories:
        if (
            not isinstance(relative, str)
            or not relative
            or PurePath(relative).is_absolute()
        ):
            raise RuntimeError("terminal tree freeze directory record is invalid")


def freeze_tree(root: Path, *, exclude_terminal_receipts: bool = False) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"cannot freeze symlink artifact: {path}")
        if path.is_file():
            if (
                exclude_terminal_receipts
                and path.parent == root
                and path.name in TERMINAL_RECEIPT_NAMES
            ):
                continue
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _verify_frozen_tree_before_terminal_publish(
    root: Path,
    contract: Mapping[str, Any],
    terminal_path: Path,
) -> None:
    if tree_freeze_contract(root) != dict(contract):
        raise RuntimeError("terminal tree freeze contract changed before publication")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"frozen output contains a symlink: {path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if path == terminal_path:
            if mode != 0:
                raise RuntimeError("terminal receipt became readable before publication")
        elif path.is_file() and mode != 0o444:
            raise RuntimeError(f"frozen output file mode changed: {path}")
        elif path.is_dir() and mode != 0o555:
            raise RuntimeError(f"frozen output directory mode changed: {path}")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise RuntimeError("frozen output root mode changed")


def publish_frozen_terminal_receipt(
    root: Path,
    terminal_name: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if terminal_name not in TERMINAL_RECEIPT_NAMES:
        raise ValueError("terminal receipt name is not authorized")
    _validate_tree_freeze_contract_semantics(contract)
    if receipt.get("artifact_freeze_contract") != dict(contract):
        raise RuntimeError("terminal receipt does not bind its tree freeze contract")
    terminal_path = root / terminal_name
    if terminal_path.exists() or terminal_path.is_symlink():
        raise FileExistsError(terminal_path)
    payload = json.dumps(
        receipt, sort_keys=True, indent=2, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(terminal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0)
    created_stat = os.fstat(descriptor)
    created_identity = (created_stat.st_dev, created_stat.st_ino)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        freeze_tree(root, exclude_terminal_receipts=True)
        _verify_frozen_tree_before_terminal_publish(root, contract, terminal_path)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        terminal_path.chmod(0o444)
        published = True
    finally:
        if not published:
            try:
                root.chmod(0o700)
                current = terminal_path.lstat()
                if (
                    stat.S_ISREG(current.st_mode)
                    and (current.st_dev, current.st_ino) == created_identity
                ):
                    terminal_path.unlink()
            except (FileNotFoundError, OSError):
                pass


def validate_published_source63_terminal_receipt(path: Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("published source63 terminal receipt is not materialized")
    if stat.S_IMODE(candidate.stat().st_mode) != 0o444:
        raise RuntimeError("published source63 terminal receipt mode is not 0444")
    terminal_sha256_before = file_sha256(candidate)
    receipt = validate_source63_terminal_receipt(
        load_json(candidate, role="published source63 terminal receipt")
    )
    if file_sha256(candidate) != terminal_sha256_before:
        raise RuntimeError("published source63 terminal receipt changed while read")
    if receipt.get("artifacts_frozen_read_only") is not True:
        raise RuntimeError("published source63 terminal is an unfrozen diagnostic")
    root = candidate.parent.resolve(strict=True)
    expected_name = receipt["terminal_receipt_name"]
    if (
        candidate.name != expected_name
        or Path(receipt["output_root"]).resolve(strict=True) != root
        or stat.S_IMODE(root.stat().st_mode) != 0o555
    ):
        raise RuntimeError("published source63 terminal path/root binding changed")
    for name in TERMINAL_RECEIPT_NAMES:
        terminal = root / name
        if name == expected_name:
            if terminal != candidate:
                raise RuntimeError("published source63 terminal inode path changed")
        elif terminal.exists() or terminal.is_symlink():
            raise RuntimeError("published source63 output has an extra terminal receipt")
    contract = receipt["artifact_freeze_contract"]
    if tree_freeze_contract(root) != contract:
        raise RuntimeError("published source63 tree differs from its freeze contract")
    contract_files = {
        record["relative_path"]: record for record in contract["files"]
    }
    for relative, record in contract_files.items():
        artifact = root / relative
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or stat.S_IMODE(artifact.stat().st_mode) != 0o444
            or artifact.stat().st_size != record["size"]
            or file_sha256(artifact) != record["sha256"]
        ):
            raise RuntimeError("published source63 frozen file changed")
    for relative in contract["directories"]:
        directory = root / relative
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or stat.S_IMODE(directory.stat().st_mode) != 0o555
        ):
            raise RuntimeError("published source63 frozen directory changed")
    if (
        stat.S_IMODE(root.stat().st_mode) != 0o555
        or stat.S_IMODE(candidate.stat().st_mode) != 0o444
        or file_sha256(candidate) != terminal_sha256_before
        or tree_freeze_contract(root) != contract
    ):
        raise RuntimeError("published source63 tree changed during verification")
    return receipt


def static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    code_root = resolve_existing_path(args.code_root, role="code root", directory=True)
    collector_root = resolve_existing_path(
        args.collector_root, role="collector root", directory=True
    )
    source_split = resolve_existing_path(
        args.source_split, role="source split", directory=False
    )
    event_spec = resolve_existing_path(args.event_spec, role="event spec", directory=False)
    output_root = resolve_new_path(args.output, role="launcher output root")
    initializer = resolve_existing_path(
        args.initializer
        or code_root / "scripts" / "initialize_smolvla_schema5_native_event_core.py",
        role="native initializer",
        directory=False,
    )
    trainer = resolve_existing_path(
        args.trainer or code_root / "scripts" / "train_openvla_etsf_counterfactual.py",
        role="counterfactual trainer",
        directory=False,
    )
    stage_runner = resolve_existing_path(
        code_root / "scripts" / "run_etsf_bound_python_stage.py",
        role="isolated bound-stage runner",
        directory=False,
    )
    split = read_frozen_split(source_split, expected_groups=args.expected_groups)
    python = python_contract(args.python_bin)
    environment_contract = canonical_environment_contract(
        gpu_index=args.gpu_index, omp_threads=args.omp_threads
    )
    probe_environment = canonical_python_environment(
        os.environ, gpu_index=args.gpu_index, omp_threads=args.omp_threads
    )
    assert_canonical_python_environment(probe_environment, environment_contract)
    runtime = probe_python_runtime(
        Path(python["invocation_path"]), probe_environment
    )
    if runtime["python_executable"] != python["resolved_path"]:
        raise RuntimeError("target interpreter resolved path differs from runtime probe")
    implementations = implementation_closure(code_root)
    expected_static_plan_sha256 = getattr(args, "expected_static_plan_sha256", None)
    parent_guard = external_suite_parent_guard(
        args, require_live=expected_static_plan_sha256 is None
    )
    gpu_observation = gpu_audit(args.gpu_index)
    gpu_identity = {
        key: gpu_observation[key] for key in ("gpu_index", "gpu_name", "gpu_uuid")
    }
    lock_path = (
        Path(args.gpu_lock).expanduser().absolute()
        if args.gpu_lock is not None
        else Path(f"/tmp/etsf_smolvla_schema5_source63_gpu{args.gpu_index}.lock")
    )
    _reject_sensitive_path_text(str(lock_path), "GPU lock")
    run_exit = (
        Path(args.run_exit).expanduser()
        if args.run_exit is not None
        else collector_root / "run.exit"
    )
    run_exit = Path(os.path.abspath(os.fspath(run_exit)))
    if run_exit.parent.resolve(strict=True) != collector_root:
        raise ValueError("run.exit must be directly inside collector root")
    if _contains_sensitive_path_component(run_exit):
        raise ValueError("run.exit path is forbidden")
    plan: dict[str, Any] = {
        "format": FORMAT,
        "status": "static_preflight_complete_waiting_no_manifest_or_hdf5_read",
        "code_root": str(code_root),
        "collector_root": str(collector_root),
        "collector_run_exit": str(run_exit),
        "collector_manifest": str(collector_root / "manifest.json"),
        "source_split": str(source_split),
        "source_split_sha256": file_sha256(source_split),
        "event_spec": str(event_spec),
        "event_spec_sha256": file_sha256(event_spec),
        "output_root": str(output_root),
        "initializer": str(initializer),
        "trainer": str(trainer),
        "stage_runner": str(stage_runner),
        "python_contract": python,
        "runtime_contract": runtime,
        "environment_contract": environment_contract,
        "implementation_files": implementations,
        "implementation_bundle_sha256": canonical_sha256(implementations),
        "split": {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
            "all_requested_seeds": split["all_requested_seeds"],
        },
        "expected_groups": args.expected_groups,
        "training_seeds": list(TRAINING_SEEDS),
        "training_steps": TRAINING_STEPS,
        "device": "cuda",
        "amp": "bf16",
        "unfreeze_semantic": True,
        "object_names": ["can"],
        "gpu_index": args.gpu_index,
        "gpu_identity": gpu_identity,
        "gpu_lock": str(lock_path),
        "omp_threads": args.omp_threads,
        "num_workers": args.num_workers,
        "poll_seconds": args.poll_seconds,
        "collector_timeout_seconds": args.collector_timeout_seconds,
        "gpu_timeout_seconds": args.gpu_timeout_seconds,
        "initializer_timeout_seconds": args.initializer_timeout_seconds,
        "training_timeout_seconds": args.training_timeout_seconds,
        "idle_confirmations_required": 2,
        "external_suite_parent_guard": parent_guard,
        "external_suite_parent_guard_contract_sha256": parent_guard.get(
            "guard_contract_sha256"
        ),
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "manifest_read_during_static_preflight": False,
        "hdf5_opened_during_static_preflight": False,
        "nonresumable_output": True,
    }
    plan["static_plan_sha256"] = canonical_sha256(plan)
    return plan


def verify_implementation_unchanged(plan: Mapping[str, Any]) -> None:
    actual = implementation_closure(Path(str(plan["code_root"])))
    if (
        actual != plan.get("implementation_files")
        or canonical_sha256(actual) != plan.get("implementation_bundle_sha256")
    ):
        raise RuntimeError("implementation changed while watcher was running")
    current_python = python_contract(
        Path(str(plan["python_contract"]["invocation_path"]))
    )
    if current_python != plan.get("python_contract"):
        raise RuntimeError("Python executable changed while watcher was running")
    environment = canonical_python_environment(
        os.environ,
        gpu_index=int(plan["gpu_index"]),
        omp_threads=int(plan["omp_threads"]),
    )
    assert_canonical_python_environment(environment, plan["environment_contract"])
    actual_runtime = probe_python_runtime(
        Path(str(plan["python_contract"]["invocation_path"])), environment
    )
    assert_runtime_matches(
        plan["runtime_contract"], actual_runtime, role="target interpreter probe"
    )
    guard = plan.get("external_suite_parent_guard")
    if isinstance(guard, Mapping) and guard.get("enabled") is True:
        script_path = Path(str(guard.get("script_path", "")))
        if (
            script_path.is_symlink()
            or not script_path.is_file()
            or file_sha256(script_path) != guard.get("script_sha256")
        ):
            raise RuntimeError("external suite parent script changed while waiting")
    gpu_observation = gpu_audit(int(plan["gpu_index"]))
    if any(
        gpu_observation.get(key) != plan.get("gpu_identity", {}).get(key)
        for key in ("gpu_index", "gpu_name", "gpu_uuid")
    ):
        raise RuntimeError("frozen RTX4090 identity changed")


def _source63_failure_phase(
    execution_phase: str,
    partial_gpu_idle_audit: Any,
    training_lifecycle: Mapping[str, Any],
) -> str:
    if execution_phase == "gpu_idle_guard_running" and (
        isinstance(partial_gpu_idle_audit, Mapping)
        and partial_gpu_idle_audit.get("status")
        == "complete_two_idle_samples_released_for_training"
    ):
        return "after_gpu_idle_release_before_training"
    if execution_phase != "training_stage_running":
        if execution_phase not in FAILURE_PHASES:
            return "before_gpu_idle_guard"
        return execution_phase
    if training_lifecycle.get("popen_reached") is True:
        return "training_process_started"
    if training_lifecycle.get("popen_attempted") is True:
        return "training_popen_attempt"
    if training_lifecycle.get("pre_popen_guard_started") is True:
        return "training_pre_popen_guard"
    return "training_runtime_probe"


def _unreaped_stage_name(
    initializer_lifecycle: Mapping[str, Any],
    training_lifecycle: Mapping[str, Any],
) -> str | None:
    for name, lifecycle in (
        ("initialize_native_event_core", initializer_lifecycle),
        ("train_source63_counterfactual_five_seed", training_lifecycle),
    ):
        if lifecycle.get("popen_attempted") is True and not (
            lifecycle.get("popen_reached") is True
            and lifecycle.get("process_reaped") is True
            and lifecycle.get("process_group_reaped") is True
        ):
            return name
    return None


def _owned_gpu_lock_release_allowed(
    *,
    gpu_lock_acquired: bool,
    release_audit: Mapping[str, Any] | None,
    initializer_lifecycle: Mapping[str, Any],
    training_lifecycle: Mapping[str, Any],
) -> bool:
    return (
        gpu_lock_acquired
        and (release_audit is None or release_audit.get("released") is not True)
        and _unreaped_stage_name(initializer_lifecycle, training_lifecycle) is None
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = static_preflight(args)
    expected_static_plan_sha256 = getattr(args, "expected_static_plan_sha256", None)
    if expected_static_plan_sha256 is not None and (
        not _is_sha256(expected_static_plan_sha256)
        or plan["static_plan_sha256"] != expected_static_plan_sha256
    ):
        raise RuntimeError("detached child static plan SHA256 differs from preflight")
    environment = canonical_python_environment(
        os.environ, gpu_index=args.gpu_index, omp_threads=args.omp_threads
    )
    assert_canonical_python_environment(environment, plan["environment_contract"])
    # A direct ``run`` invocation that omitted ``-I`` or retained an ambient
    # alternate Torch fails before creating its output root.  ``detach`` always
    # launches this process with both the isolated flag and canonical env.
    assert_runtime_matches(
        plan["runtime_contract"],
        current_python_runtime(),
        role="watcher process",
    )
    activate_trusted_scripts_path(Path(plan["code_root"]))
    output = Path(plan["output_root"])
    output.mkdir(mode=0o700)
    state_path = output / "launch_state.json"
    plan_path = output / "launch_plan.json"
    lock_path = Path(str(plan["gpu_lock"]))
    token = canonical_sha256(
        {"pid": os.getpid(), "plan": plan["static_plan_sha256"], "time": time.time_ns()}
    )
    lock_payload = {"format": FORMAT, "pid": os.getpid(), "token": token}
    immutable_json_new(plan_path, plan)
    acquire_lock(output / "launch.lock", lock_payload)
    state: dict[str, Any] = {
        "format": STATE_FORMAT,
        "status": "starting",
        "pid": os.getpid(),
        "static_plan_sha256": plan["static_plan_sha256"],
        "stage_results": {},
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
        "manifest_read": False,
        "hdf5_opened": False,
        "test_labels_used": False,
        "test_hdf_label_datasets_opened": 0,
    }
    atomic_json(state_path, state)
    gpu_lock_acquired = False
    release_audit: dict[str, Any] | None = None
    execution_phase = "before_gpu_idle_guard"
    initializer_lifecycle: dict[str, Any] = {}
    training_lifecycle: dict[str, Any] = {}
    training_stage_returned = False
    try:
        acquire_lock(lock_path, lock_payload)
        gpu_lock_acquired = True
        exit_audit = wait_for_collector_exit(
            Path(plan["collector_run_exit"]),
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.collector_timeout_seconds,
        )
        verify_implementation_unchanged(plan)
        state.update(
            {
                "status": "validating_completed_manifest_without_hdf5",
                "collector_exit_audit": exit_audit,
                "manifest_read": True,
                "hdf5_opened": False,
            }
        )
        atomic_json(state_path, state)
        split = read_frozen_split(
            Path(plan["source_split"]), expected_groups=args.expected_groups
        )
        metadata = validate_collector_metadata(
            Path(plan["collector_manifest"]),
            split=split,
            event_spec_sha256=str(plan["event_spec_sha256"]),
            expected_groups=args.expected_groups,
        )
        state["completed_manifest_audit"] = {
            key: metadata[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "requested_seeds",
                "resolved_seeds",
                "modeling_sha256",
                "bridge_sha256",
                "event_spec_sha256",
            )
        }
        state["status"] = "self_validating_completed_hdf5_groups"
        state["hdf5_opened"] = True
        atomic_json(state_path, state)
        group_audits = audit_group_files(Path(plan["collector_root"]), metadata)
        if len(group_audits) != args.expected_groups:
            raise RuntimeError("completed source group audit count changed")
        state["group_audit_sha256"] = canonical_sha256(group_audits)
        state["group_audit_count"] = len(group_audits)
        state["train_validation_hdf_self_validated"] = sum(
            audit["split"] in ("train", "validation") for audit in group_audits
        )
        state["test_hdf_byte_hashed"] = sum(
            audit["split"] == "test" for audit in group_audits
        )
        state["test_labels_used"] = False
        state["test_hdf_label_datasets_opened"] = 0
        state["test_hdf_identity_attrs_opened_by_watcher"] = 0
        state["status"] = "freezing_independent_source_snapshot"
        atomic_json(state_path, state)
        snapshot = freeze_source_snapshot(
            output,
            collector_exit=Path(plan["collector_run_exit"]),
            manifest_path=Path(plan["collector_manifest"]),
            event_spec=Path(plan["event_spec"]),
            source_split=Path(plan["source_split"]),
            metadata=metadata,
            group_audits=group_audits,
        )
        commands = build_stage_commands(
            python_bin=Path(plan["python_contract"]["invocation_path"]),
            stage_runner=Path(plan["stage_runner"]),
            launch_plan=plan_path,
            static_plan_sha256=str(plan["static_plan_sha256"]),
            runtime_contract_sha256=str(
                plan["runtime_contract"]["runtime_contract_sha256"]
            ),
            initializer=Path(plan["initializer"]),
            trainer=Path(plan["trainer"]),
            output_root=output,
            snapshot_receipt=snapshot,
            modeling_sha256=str(metadata["modeling_sha256"]),
            bridge_sha256=str(metadata["bridge_sha256"]),
            num_workers=args.num_workers,
        )
        commands[1]["external_suite_parent_guard_contract_sha256"] = plan[
            "external_suite_parent_guard_contract_sha256"
        ]
        execution_plan: dict[str, Any] = {
            "format": FORMAT,
            "status": "post_collection_execution_plan_frozen",
            "static_plan_sha256": plan["static_plan_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "commands": commands,
            "execution_order": [command["stage"] for command in commands],
            "runtime_contract_sha256": plan["runtime_contract"][
                "runtime_contract_sha256"
            ],
            "environment_contract_sha256": plan["environment_contract"][
                "environment_contract_sha256"
            ],
            "external_suite_parent_guard": plan["external_suite_parent_guard"],
            "external_suite_parent_guard_contract_sha256": plan[
                "external_suite_parent_guard_contract_sha256"
            ],
            "fresh_inputs_accepted": False,
            "fresh_labels_read": False,
            "test_labels_used": False,
            "test_hdf_label_datasets_opened": 0,
        }
        execution_plan["execution_plan_sha256"] = canonical_sha256(execution_plan)
        immutable_json_new(output / "execution_plan.json", execution_plan)
        state["snapshot_sha256"] = snapshot["snapshot_sha256"]
        state["execution_plan_sha256"] = execution_plan["execution_plan_sha256"]
        verify_implementation_unchanged(plan)
        initialize_result = run_subprocess_stage(
            commands[0],
            output_root=output,
            environment=environment,
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.initializer_timeout_seconds,
            expected_runtime_contract=plan["runtime_contract"],
            lifecycle=initializer_lifecycle,
        )
        initialize_result["artifact_audit"] = validate_initialized_output(
            Path(commands[0]["output"])
        )
        atomic_json(
            output / "stage_receipts" / f"{commands[0]['stage']}.json",
            initialize_result,
        )
        state["stage_results"][commands[0]["stage"]] = initialize_result
        state["current_stage"] = None
        atomic_json(state_path, state)
        verify_implementation_unchanged(plan)
        execution_phase = "gpu_idle_guard_running"
        state["gpu_idle_audit"] = wait_for_idle_4090(
            gpu_index=args.gpu_index,
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.gpu_timeout_seconds,
            external_parent_guard=plan["external_suite_parent_guard"],
            expected_gpu_identity=plan["gpu_identity"],
        )
        execution_phase = "after_gpu_idle_release_before_training"
        verify_implementation_unchanged(plan)
        commands[1]["gpu_idle_release_audit_sha256"] = state["gpu_idle_audit"][
            "gpu_idle_release_audit_sha256"
        ]
        atomic_json(state_path, state)
        execution_phase = "training_stage_running"
        training_result = run_subprocess_stage(
            commands[1],
            output_root=output,
            environment=environment,
            state=state,
            state_path=state_path,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.training_timeout_seconds,
            expected_runtime_contract=plan["runtime_contract"],
            pre_popen_guard=lambda: audit_training_start_guard(
                plan, state["gpu_idle_audit"]
            ),
            lifecycle=training_lifecycle,
        )
        training_stage_returned = True
        execution_phase = "after_training_stage_before_gpu_lock_release"
        training_start_guard_audit = training_result.get("pre_popen_guard_audit")
        if not isinstance(training_start_guard_audit, Mapping):
            raise RuntimeError("training stage lacks its pre-Popen guard audit")
        training_start_guard_audit = dict(training_start_guard_audit)
        training_result["artifact_audit"] = validate_training_output(
            Path(commands[1]["output"]),
            expected_pretrained_sha256=file_sha256(Path(commands[0]["output"])),
        )
        training_result["external_suite_parent_guard_contract_sha256"] = plan[
            "external_suite_parent_guard_contract_sha256"
        ]
        release_audit = release_owned_lock(lock_path, token, strict=True)
        execution_phase = "after_gpu_lock_release_before_terminal_finalize"
        training_result["gpu_idle_release_audit_sha256"] = state[
            "gpu_idle_audit"
        ]["gpu_idle_release_audit_sha256"]
        training_result["training_start_guard_audit_sha256"] = (
            training_start_guard_audit["training_start_guard_audit_sha256"]
        )
        training_result["gpu_lock_release_audit_sha256"] = release_audit[
            "release_audit_sha256"
        ]
        training_result = sign_canonical_receipt(
            training_result, field="stage_receipt_sha256"
        )
        atomic_json(
            output / "stage_receipts" / f"{commands[1]['stage']}.json",
            training_result,
        )
        state["stage_results"][commands[1]["stage"]] = training_result
        state["current_stage"] = None
        state["gpu_lock_release_audit"] = release_audit
        state["status"] = TERMINAL_STATUS
        state["target_data_read"] = False
        state["target_labels_read"] = False
        state["test_labels_used"] = False
        state["test_hdf_label_datasets_opened"] = 0
        state["test_hdf_identity_attrs_opened"] = 5
        state["finished_unix"] = time.time()
        atomic_json(state_path, state)
        inventory = recursive_artifact_inventory(output)
        atomic_json(output / "artifact_inventory.json", inventory)
        freeze_contract = tree_freeze_contract(output)
        final_receipt = sign_canonical_receipt({
            "format": FORMAT,
            "status": TERMINAL_STATUS,
            "output_root": str(output),
            "terminal_receipt_name": "final_receipt.json",
            "launcher_pid": os.getpid(),
            "gpu_lock_path": str(lock_path),
            "gpu_lock_token_sha256": hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest(),
            "gpu_identity": plan["gpu_identity"],
            "static_plan_sha256": plan["static_plan_sha256"],
            "execution_plan_sha256": execution_plan["execution_plan_sha256"],
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "artifact_inventory_sha256": inventory["inventory_sha256"],
            "initialized_checkpoint_sha256": file_sha256(Path(commands[0]["output"])),
            "training_audit": training_result["artifact_audit"],
            "training_stage_receipt": training_result,
            "training_stage_receipt_sha256": training_result[
                "stage_receipt_sha256"
            ],
            "gpu_idle_audit": state["gpu_idle_audit"],
            "gpu_idle_release_audit_sha256": state["gpu_idle_audit"][
                "gpu_idle_release_audit_sha256"
            ],
            "training_start_guard_audit": training_start_guard_audit,
            "training_start_guard_audit_sha256": training_start_guard_audit[
                "training_start_guard_audit_sha256"
            ],
            "external_suite_parent_guard": plan["external_suite_parent_guard"],
            "external_suite_parent_guard_contract_sha256": plan[
                "external_suite_parent_guard_contract_sha256"
            ],
            "gpu_lock_release_audit": release_audit,
            "gpu_lock_release_audit_sha256": release_audit[
                "release_audit_sha256"
            ],
            "target_data_read": False,
            "target_labels_read": False,
            "fresh_inputs_accepted": False,
            "fresh_labels_read": False,
            "test_labels_used": False,
            "test_hdf_label_datasets_opened": 0,
            "test_hdf_identity_attrs_opened": 5,
            "artifacts_frozen_read_only": True,
            "artifact_freeze_contract": freeze_contract,
            "artifact_freeze_contract_sha256": freeze_contract[
                "tree_freeze_contract_sha256"
            ],
        })
        validate_source63_terminal_receipt(final_receipt)
        publish_frozen_terminal_receipt(
            output, "final_receipt.json", final_receipt, freeze_contract
        )
        return final_receipt
    except BaseException as error:
        partial_gpu_idle_audit = state.get("gpu_idle_audit")
        failed_from_status = _source63_failure_phase(
            execution_phase, partial_gpu_idle_audit, training_lifecycle
        )
        lock_released_before_failure = (
            isinstance(release_audit, Mapping)
            and release_audit.get("released") is True
        )
        unreaped_stage = _unreaped_stage_name(
            initializer_lifecycle, training_lifecycle
        )
        if training_lifecycle.get("popen_reached") is not True:
            training_group_binding_status = (
                "popen_attempt_unproven"
                if training_lifecycle.get("popen_attempted") is True
                else "not_reached"
            )
        elif (
            training_lifecycle.get("process_group_isolated") is True
            and training_lifecycle.get("process_group_id")
            == training_lifecycle.get("process_pid")
        ):
            training_group_binding_status = "bound_isolated"
        else:
            training_group_binding_status = "failed_unproven"
        if _owned_gpu_lock_release_allowed(
            gpu_lock_acquired=gpu_lock_acquired,
            release_audit=release_audit,
            initializer_lifecycle=initializer_lifecycle,
            training_lifecycle=training_lifecycle,
        ):
            release_audit = release_owned_lock(lock_path, token, strict=False)
        state.update(
            {
                "status": FAILURE_STATUS,
                "error_type": type(error).__name__,
                "error": str(error),
                "target_data_read": False,
                "target_labels_read": False,
                "fresh_inputs_accepted": False,
                "fresh_labels_read": False,
                "test_labels_used": False,
                "test_hdf_label_datasets_opened": 0,
                "finished_unix": time.time(),
                "gpu_lock_release_audit": release_audit,
            }
        )
        try:
            atomic_json(state_path, state)
            failure_freeze_contract = (
                tree_freeze_contract(output) if unreaped_stage is None else None
            )
            failure_receipt = sign_canonical_receipt(
                {
                    "format": FORMAT,
                    "status": FAILURE_STATUS,
                    "output_root": str(output),
                    "terminal_receipt_name": "failure_receipt.json",
                    "launcher_pid": os.getpid(),
                    "gpu_lock_path": str(lock_path),
                    "gpu_lock_token_sha256": hashlib.sha256(
                        token.encode("utf-8")
                    ).hexdigest(),
                    "gpu_identity": plan["gpu_identity"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "static_plan_sha256": plan["static_plan_sha256"],
                    "external_suite_parent_guard": plan[
                        "external_suite_parent_guard"
                    ],
                    "external_suite_parent_guard_contract_sha256": plan[
                        "external_suite_parent_guard_contract_sha256"
                    ],
                    "failure_phase": failed_from_status,
                    "gpu_idle_guard_started": partial_gpu_idle_audit is not None,
                    "gpu_idle_release_reached": (
                        isinstance(partial_gpu_idle_audit, Mapping)
                        and partial_gpu_idle_audit.get("status")
                        == "complete_two_idle_samples_released_for_training"
                    ),
                    "training_pre_popen_guard_started": training_lifecycle.get(
                        "pre_popen_guard_started"
                    ) is True,
                    "training_pre_popen_guard_completed": training_lifecycle.get(
                        "pre_popen_guard_completed"
                    ) is True,
                    "training_popen_attempted": training_lifecycle.get(
                        "popen_attempted"
                    ) is True,
                    "training_popen_reached": training_lifecycle.get(
                        "popen_reached"
                    ) is True,
                    "training_process_pid": training_lifecycle.get("process_pid"),
                    "training_process_reaped": training_lifecycle.get(
                        "process_reaped"
                    ) is True,
                    "training_process_group_id": training_lifecycle.get(
                        "process_group_id"
                    ),
                    "training_process_group_isolated": training_lifecycle.get(
                        "process_group_isolated"
                    ) is True,
                    "training_process_group_reaped": training_lifecycle.get(
                        "process_group_reaped"
                    ) is True,
                    "training_process_group_binding_status": (
                        training_group_binding_status
                    ),
                    "training_stage_returned": training_stage_returned,
                    "gpu_lock_acquired": gpu_lock_acquired,
                    "gpu_lock_released": (
                        isinstance(release_audit, Mapping)
                        and release_audit.get("released") is True
                    ),
                    "gpu_lock_released_before_failure": lock_released_before_failure,
                    "unreaped_stage_process": unreaped_stage,
                    "gpu_lock_retained_for_unreaped_stage_process": (
                        unreaped_stage is not None
                        and gpu_lock_acquired
                        and not (
                            isinstance(release_audit, Mapping)
                            and release_audit.get("released") is True
                        )
                    ),
                    "partial_gpu_idle_guard_audit": partial_gpu_idle_audit,
                    "gpu_lock_release_audit": release_audit,
                    "target_data_read": False,
                    "target_labels_read": False,
                    "fresh_inputs_accepted": False,
                    "fresh_labels_read": False,
                    "test_labels_used": False,
                    "test_hdf_label_datasets_opened": 0,
                    "artifacts_frozen_read_only": unreaped_stage is None,
                    "artifact_freeze_contract": failure_freeze_contract,
                    "artifact_freeze_contract_sha256": (
                        failure_freeze_contract["tree_freeze_contract_sha256"]
                        if failure_freeze_contract is not None
                        else None
                    ),
                }
            )
            validate_source63_terminal_receipt(failure_receipt)
            if unreaped_stage is None:
                publish_frozen_terminal_receipt(
                    output,
                    "failure_receipt.json",
                    failure_receipt,
                    failure_freeze_contract,
                )
            else:
                immutable_json_new(output / "failure_receipt.json", failure_receipt)
        except BaseException:
            pass
        raise
    finally:
        if _owned_gpu_lock_release_allowed(
            gpu_lock_acquired=gpu_lock_acquired,
            release_audit=release_audit,
            initializer_lifecycle=initializer_lifecycle,
            training_lifecycle=training_lifecycle,
        ):
            release_owned_lock(lock_path, token, strict=False)


def common_run_argv(
    args: argparse.Namespace, *, expected_static_plan_sha256: str
) -> list[str]:
    script = Path(__file__).resolve()
    argv = [
        str(args.python_bin),
        "-I",
        str(script),
        "run",
        "--expected-static-plan-sha256",
        expected_static_plan_sha256,
        "--code-root",
        str(args.code_root),
        "--collector-root",
        str(args.collector_root),
        "--source-split",
        str(args.source_split),
        "--event-spec",
        str(args.event_spec),
        "--output",
        str(args.output),
        "--python-bin",
        str(args.python_bin),
        "--gpu-index",
        str(args.gpu_index),
        "--poll-seconds",
        str(args.poll_seconds),
        "--collector-timeout-seconds",
        str(args.collector_timeout_seconds),
        "--gpu-timeout-seconds",
        str(args.gpu_timeout_seconds),
        "--initializer-timeout-seconds",
        str(args.initializer_timeout_seconds),
        "--training-timeout-seconds",
        str(args.training_timeout_seconds),
        "--expected-groups",
        str(args.expected_groups),
        "--num-workers",
        str(args.num_workers),
        "--omp-threads",
        str(args.omp_threads),
    ]
    for flag, value in (
        ("--run-exit", args.run_exit),
        ("--initializer", args.initializer),
        ("--trainer", args.trainer),
        ("--gpu-lock", args.gpu_lock),
        ("--external-suite-parent-pid", getattr(args, "external_suite_parent_pid", None)),
        (
            "--external-suite-parent-start-ticks",
            getattr(args, "external_suite_parent_start_ticks", None),
        ),
        (
            "--external-suite-parent-boot-id",
            getattr(args, "external_suite_parent_boot_id", None),
        ),
        (
            "--external-suite-parent-cmdline-sha256",
            getattr(args, "external_suite_parent_cmdline_sha256", None),
        ),
        (
            "--external-suite-parent-script",
            getattr(args, "external_suite_parent_script", None),
        ),
        (
            "--external-suite-parent-script-sha256",
            getattr(args, "external_suite_parent_script_sha256", None),
        ),
    ):
        if value is not None:
            argv.extend([flag, str(value)])
    return argv


def detach(args: argparse.Namespace) -> dict[str, Any]:
    # Static preflight intentionally does not read manifest/HDF5 and also proves
    # the output root is absent before the background process is created.
    plan = static_preflight(args)
    output = Path(plan["output_root"])
    receipt_path = resolve_new_path(
        args.detach_receipt
        or output.parent / f"{output.name}.detach_receipt.json",
        role="detach receipt",
    )
    daemon_log = resolve_new_path(
        args.detach_log or output.parent / f"{output.name}.launcher.log",
        role="detached launcher log",
    )
    argv = common_run_argv(
        args, expected_static_plan_sha256=str(plan["static_plan_sha256"])
    )
    environment = canonical_python_environment(
        os.environ, gpu_index=args.gpu_index, omp_threads=args.omp_threads
    )
    assert_canonical_python_environment(environment, plan["environment_contract"])
    with daemon_log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=environment,
        )
    receipt: dict[str, Any] = {
        "format": DETACH_FORMAT,
        "status": "detached_server_side_watcher_started",
        "pid": process.pid,
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "output_root": str(output),
        "daemon_log": str(daemon_log),
        "static_plan_sha256": plan["static_plan_sha256"],
        "runtime_contract_sha256": plan["runtime_contract"][
            "runtime_contract_sha256"
        ],
        "environment_contract_sha256": plan["environment_contract"][
            "environment_contract_sha256"
        ],
        "external_suite_parent_guard": plan["external_suite_parent_guard"],
        "watcher_isolated_python": True,
        "survives_client_disconnect": True,
        "manifest_read_by_detach": False,
        "hdf5_opened_by_detach": False,
        "fresh_inputs_accepted": False,
        "fresh_labels_read": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    immutable_json_new(receipt_path, receipt)
    print("SMOLVLA_SOURCE63_DETACHED=" + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--code-root", type=Path, default=root)
    parser.add_argument("--collector-root", type=Path, required=True)
    parser.add_argument("--run-exit", type=Path)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--external-suite-parent-pid", type=int)
    parser.add_argument("--external-suite-parent-start-ticks", type=int)
    parser.add_argument("--external-suite-parent-boot-id")
    parser.add_argument("--external-suite-parent-cmdline-sha256")
    parser.add_argument("--external-suite-parent-script", type=Path)
    parser.add_argument("--external-suite-parent-script-sha256")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--collector-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--gpu-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--initializer-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--training-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--expected-groups", type=int, default=EXPECTED_GROUPS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--omp-threads", type=int, default=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Authenticate static inputs without reading manifest/HDF5 or launching.",
    )
    add_common_arguments(preflight_parser)
    run_parser = subparsers.add_parser("run", help="Run watcher in the foreground.")
    add_common_arguments(run_parser)
    run_parser.add_argument("--expected-static-plan-sha256")
    detach_parser = subparsers.add_parser(
        "detach", help="Start a new-session server-side watcher and return a receipt."
    )
    add_common_arguments(detach_parser)
    detach_parser.add_argument("--detach-receipt", type=Path)
    detach_parser.add_argument("--detach-log", type=Path)
    args = parser.parse_args()
    if (
        args.gpu_index < 0
        or not 0 < args.poll_seconds <= 60
        or args.collector_timeout_seconds < 0
        or args.gpu_timeout_seconds < 0
        or args.initializer_timeout_seconds <= 0
        or args.training_timeout_seconds < 0
        or args.expected_groups != EXPECTED_GROUPS
        or args.num_workers < 0
        or args.omp_threads <= 0
    ):
        parser.error("invalid source63 watcher timing/count/device arguments")
    if args.command == "run" and (
        args.expected_static_plan_sha256 is not None
        and not _is_sha256(args.expected_static_plan_sha256)
    ):
        parser.error("invalid expected static plan SHA256")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        plan = static_preflight(args)
        print("SMOLVLA_SOURCE63_STATIC_PREFLIGHT=" + json.dumps(plan, sort_keys=True))
        return
    if args.command == "detach":
        detach(args)
        return
    result = execute(args)
    print("SMOLVLA_SOURCE63_TRAINING_COMPLETE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
