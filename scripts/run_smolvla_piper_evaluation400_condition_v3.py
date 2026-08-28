#!/usr/bin/env python3
"""Execute exactly one bound evaluation400 condition.

The runner owns simulator interaction, not orchestration.  It resets the exact
frozen identity, generates the same four SmolVLA root candidates, selects only
the condition's root candidate, and thereafter always executes the lowest
legal continuation.  Simulator ``info.success`` is the sole task outcome.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import importlib.abc
import importlib.machinery
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


REQUEST_FORMAT = "etsf_smolvla_piper_evaluation400_condition_request_v3"
RESULT_FORMAT = "etsf_smolvla_piper_evaluation400_condition_runner_result_v3"
RESULT_STATUS = "complete_single_condition_from_bound_snapshot"
CONTINUATION_CONTRACT = "frozen_lowest_legal_feasibility_continuation_v1"
CANDIDATE_COUNT = 4
ACTION_DIM = 14
SHA_CHARS = frozenset("0123456789abcdef")
DEPENDENCY_CLOSURE_FORMAT = (
    "etsf_smolvla_piper_evaluation400_local_import_dependency_closure_v3"
)
DEPENDENCY_CLOSURE_STATUS = (
    "recursive_static_imports_and_limited_dynamic_imports_content_addressed"
)
BOUND_EXTERNAL_DYNAMIC_IMPORTS = frozenset({
    "rlinf.envs.robotwin.robotwin_env",
    "robotwin.envs.vector_env",
    "envs._base_task",
    "envs.robot.robot",
    "envs.move_can_pot",
})
SOURCE_RANK_NUMERIC_CONTRACT = (
    "ieee754_float32_training_order_base_plus_residual_div_temperature"
)


class ConditionRunnerError(RuntimeError):
    """A request, dependency, selector, reset, or simulator invariant failed."""


class ConditionBackend(Protocol):
    max_steps: int
    continuation_policy_sha256: str

    def reset(self, requested_seed: int) -> tuple[Any, Mapping[str, Any]]: ...
    def query(self, observation: Any, query_index: int) -> Mapping[str, Any]: ...
    def select_etsf(self, query: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]: ...
    def step(self, action: np.ndarray) -> tuple[Any, bool, bool, Mapping[str, Any]]: ...
    def snapshot(self) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ConditionRunnerError("dependency is not a regular file")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def bound_file(path: Path, expected_sha256: str, role: str) -> Path:
    source = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if source.is_symlink() or not source.is_file() or source.stat().st_mode & 0o222:
        raise ConditionRunnerError(f"{role} must be a frozen regular file")
    resolved = source.resolve(strict=True)
    if not is_sha(expected_sha256) or file_sha256(resolved) != expected_sha256:
        raise ConditionRunnerError(f"{role} file SHA mismatch")
    return resolved


def read_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    source = bound_file(path, expected_sha256, role)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConditionRunnerError(f"{role} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ConditionRunnerError(f"{role} must contain an object")
    return value


def _verify_logical(value: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = dict(value)
    logical = unsigned.pop(field, None)
    if not is_sha(logical) or logical != canonical_sha256(unsigned):
        raise ConditionRunnerError(f"{role} logical SHA mismatch")
    return str(logical)


def load_bound_module(path: Path, expected_sha256: str, role: str) -> Any:
    source = bound_file(path, expected_sha256, role)
    module_name = f"_etsf_eval400_{role.replace(' ', '_')}_{expected_sha256[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ConditionRunnerError(f"cannot load {role}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _dotted_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if not isinstance(node, ast.Attribute):
        return ""
    parts: list[str] = []
    cursor: ast.AST = node
    while isinstance(cursor, ast.Attribute):
        parts.append(cursor.attr)
        cursor = cursor.value
    if isinstance(cursor, ast.Name):
        parts.append(cursor.id)
        return ".".join(reversed(parts))
    return ""


class _ClosedLocalImportFinder(importlib.abc.MetaPathFinder):
    def __init__(self, scripts_root: Path, allowed_modules: set[str]) -> None:
        self.scripts_root = scripts_root
        self.allowed_modules = allowed_modules

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        top_level = fullname.split(".", 1)[0]
        local_candidate = self.scripts_root / f"{top_level}.py"
        if not local_candidate.is_file():
            return None
        if top_level not in self.allowed_modules:
            raise ImportError(f"local module is outside frozen closure: {top_level}")
        if "." in fullname:
            return None
        return importlib.machinery.PathFinder.find_spec(
            fullname, [str(self.scripts_root)]
        )


def install_local_dependency_closure(
    *, path: Path, file_sha256_expected: str, logical_sha256_expected: str,
    expected_roots: Sequence[Path],
) -> dict[str, Any]:
    """Recompute the recursive local import closure, then install a deny guard."""

    value = read_json(path, file_sha256_expected, "local dependency closure")
    logical = _verify_logical(value, "closure_sha256", "local dependency closure")
    expected_fields = {
        "format", "status", "scripts_root", "roots", "files",
        "dynamic_imports", "unclosed_local_import_count",
        "unbounded_dynamic_import_count", "closure_sha256",
    }
    scripts_root = Path(str(value.get("scripts_root")))
    if (
        set(value) != expected_fields
        or value.get("format") != DEPENDENCY_CLOSURE_FORMAT
        or value.get("status") != DEPENDENCY_CLOSURE_STATUS
        or logical != logical_sha256_expected
        or not scripts_root.is_absolute()
        or scripts_root.is_symlink()
        or not scripts_root.is_dir()
        or type(value.get("unclosed_local_import_count")) is not int
        or value["unclosed_local_import_count"] != 0
        or type(value.get("unbounded_dynamic_import_count")) is not int
        or value["unbounded_dynamic_import_count"] != 0
    ):
        raise ConditionRunnerError("local dependency closure header changed")
    file_rows = value.get("files")
    root_rows = value.get("roots")
    dynamic_rows = value.get("dynamic_imports")
    if (
        not isinstance(file_rows, list)
        or not isinstance(root_rows, list)
        or not isinstance(dynamic_rows, list)
    ):
        raise ConditionRunnerError("local dependency closure rows changed")
    allowed: dict[str, str] = {}
    for row in file_rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "file_sha256"}
            or not isinstance(row.get("relative_path"), str)
            or "/" in row["relative_path"]
            or not row["relative_path"].endswith(".py")
            or not is_sha(row.get("file_sha256"))
            or row["relative_path"] in allowed
        ):
            raise ConditionRunnerError("local dependency closure file row changed")
        allowed[row["relative_path"]] = str(row["file_sha256"])
    expected_relative_roots: list[str] = []
    for raw in expected_roots:
        source = bound_file(raw, file_sha256(raw), "expected local closure root")
        try:
            relative = source.relative_to(scripts_root).as_posix()
        except ValueError as error:
            raise ConditionRunnerError("expected closure root escaped scripts root") from error
        expected_relative_roots.append(relative)
    if root_rows != sorted(set(expected_relative_roots)):
        raise ConditionRunnerError("local dependency closure roots changed")

    pending = list(root_rows)
    reached: set[str] = set()
    observed_dynamic: list[dict[str, Any]] = []
    while pending:
        relative = pending.pop()
        if relative in reached:
            continue
        if relative not in allowed:
            raise ConditionRunnerError("local import root is absent from closure")
        source = bound_file(
            scripts_root / relative, allowed[relative], f"closed local module {relative}"
        )
        reached.add(relative)
        try:
            tree = ast.parse(source.read_bytes(), filename=str(source))
        except (SyntaxError, ValueError) as error:
            raise ConditionRunnerError("closed local module cannot be parsed") from error
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ConditionRunnerError("relative local import is forbidden")
                modules = [node.module] if node.module else []
            for module in modules:
                target = f"{module.split('.', 1)[0]}.py"
                if target in allowed:
                    pending.append(target)
                elif (scripts_root / target).is_file():
                    raise ConditionRunnerError(
                        f"local import is missing from frozen closure: {target}"
                    )
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_call_name(node.func)
            if dotted == "__import__":
                raise ConditionRunnerError("unbounded __import__ is forbidden")
            if dotted == "importlib.import_module":
                argument = node.args[0] if node.args else None
                if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                    raise ConditionRunnerError("nonliteral dynamic import is forbidden")
                module = argument.value
                target = f"{module.split('.', 1)[0]}.py"
                if target in allowed:
                    pending.append(target)
                elif module not in BOUND_EXTERNAL_DYNAMIC_IMPORTS:
                    raise ConditionRunnerError("dynamic import lacks reviewed authority")
                else:
                    observed_dynamic.append({
                        "relative_path": relative,
                        "line": node.lineno,
                        "kind": "literal_external_runtime_module",
                        "module": module,
                        "authority": "runtime_contract.runtime_source_artifacts",
                    })
            elif dotted == "importlib.util.spec_from_file_location":
                observed_dynamic.append({
                    "relative_path": relative,
                    "line": node.lineno,
                    "kind": "content_addressed_file_module",
                    "module": None,
                    "authority": "condition_runner.bound_file_path_and_sha256",
                })
    if reached != set(allowed):
        raise ConditionRunnerError("dependency closure contains unreachable local code")
    observed_dynamic.sort(
        key=lambda row: (
            row["relative_path"], row["line"], row["kind"], row.get("module") or ""
        )
    )
    if observed_dynamic != dynamic_rows:
        raise ConditionRunnerError("dynamic import authority closure changed")
    allowed_modules = {Path(relative).stem for relative in allowed}
    sys.meta_path.insert(0, _ClosedLocalImportFinder(scripts_root, allowed_modules))
    sys.path.insert(0, str(scripts_root))
    return value


REQUEST_FIELDS = {
    "format", "status", "plan_sha256", "bundle_sha256", "claim_sha256",
    "pair_id", "ordinal", "requested_seed", "resolved_seed",
    "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
    "initial_commanded_drive_target_sha256", "attempt", "pair_identity_sha256",
    "condition", "condition_ordinal", "condition_order", "shared_snapshot_sha256",
    "candidate_count", "candidate_generation_contract_sha256",
    "postfreeze_identity_or_order_change_authorized",
    "outcome_visible_before_condition_start", "request_sha256",
}


def validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    logical = _verify_logical(value, "request_sha256", "condition request")
    if (
        set(value) != REQUEST_FIELDS
        or value.get("format") != REQUEST_FORMAT
        or value.get("status") != "write_ahead_before_condition_popen"
        or value.get("condition") not in {"baseline", "etsf"}
        or type(value.get("condition_ordinal")) is not int
        or value["condition_ordinal"] not in (0, 1)
        or not isinstance(value.get("condition_order"), list)
        or len(value["condition_order"]) != 2
        or set(value["condition_order"]) != {"baseline", "etsf"}
        or value["condition_order"][value["condition_ordinal"]] != value["condition"]
        or type(value.get("ordinal")) is not int or value["ordinal"] < 0
        or type(value.get("attempt")) is not int or value["attempt"] != 0
        or type(value.get("requested_seed")) is not int
        or type(value.get("resolved_seed")) is not int
        or value["requested_seed"] != value["resolved_seed"]
        or type(value.get("candidate_count")) is not int
        or value["candidate_count"] != CANDIDATE_COUNT
        or value.get("postfreeze_identity_or_order_change_authorized") is not False
        or value.get("outcome_visible_before_condition_start") is not False
        or any(
            not is_sha(value.get(field))
            for field in (
                "plan_sha256", "bundle_sha256", "claim_sha256", "pair_id",
                "pair_identity_sha256", "shared_snapshot_sha256",
                "candidate_generation_contract_sha256",
                "initial_scene_state_sha256", "initial_measured_joint_state_sha256",
                "initial_commanded_drive_target_sha256",
            )
        )
        or logical != value["request_sha256"]
    ):
        raise ConditionRunnerError("condition request contract changed")
    return dict(value)


def _query_contract(query: Mapping[str, Any]) -> tuple[list[str], list[bool], int]:
    native = query.get("native_action_sha256")
    mask = np.asarray(query.get("feasibility_mask"), dtype=bool)
    actions = np.asarray(query.get("mapped_actions"), dtype=np.float32)
    lowest = query.get("lowest_legal_original_candidate_index")
    if (
        not isinstance(native, list) or len(native) != CANDIDATE_COUNT
        or any(not is_sha(value) for value in native) or len(set(native)) != CANDIDATE_COUNT
        or mask.shape != (CANDIDATE_COUNT,) or not bool(mask.any())
        or actions.shape[0] != CANDIDATE_COUNT or actions.shape[2] != ACTION_DIM
        or not np.isfinite(actions).all()
        or type(lowest) is not int or lowest != int(np.flatnonzero(mask)[0])
    ):
        raise ConditionRunnerError("four-candidate query contract changed")
    return list(native), mask.tolist(), int(lowest)


def apply_deployment_prediction_scales(
    output: Mapping[str, Any], *, object_mean: Any, object_std: Any,
    duration_scale_multiplier: float, object_scale_multiplier: float,
) -> dict[str, Any]:
    """Convert object uncertainty to physical XYZ and apply frozen scales."""

    for role, value in (
        ("duration", duration_scale_multiplier),
        ("object", object_scale_multiplier),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ConditionRunnerError(f"{role} deployment scale is invalid")
    try:
        if (
            object_mean.ndim != 1
            or object_std.shape != object_mean.shape
            or not bool((object_std > 0).all())
            or output["object_delta_mean"].shape[-1] != object_mean.shape[0]
            or output["object_delta_log_scale"].shape
            != output["object_delta_mean"].shape
        ):
            raise ConditionRunnerError("object normalization shape changed")
        return {
            "duration_log_mean": output["duration_selected_log_mean"],
            "duration_log_scale": output["duration_selected_log_scale"]
            + math.log(float(duration_scale_multiplier)),
            "object_mean": output["object_delta_mean"] * object_std + object_mean,
            "object_log_scale": output["object_delta_log_scale"]
            + object_std.log()
            + math.log(float(object_scale_multiplier)),
        }
    except (AttributeError, KeyError, TypeError) as error:
        raise ConditionRunnerError("deployment prediction scale inputs changed") from error


def execute_condition(
    *, request: Mapping[str, Any], backend: ConditionBackend,
) -> tuple[dict[str, Any], bytes, bytes]:
    request = validate_request(request)
    if (
        type(backend.max_steps) is not int
        or backend.max_steps != 200
        or not is_sha(getattr(backend, "schema6_execution_authority_file_sha256", None))
        or not is_sha(getattr(backend, "schema6_runtime_contract_sha256", None))
    ):
        raise ConditionRunnerError("condition backend is not exact 200-step runtime")
    root_observation, identity = backend.reset(int(request["requested_seed"]))
    expected_identity = {
        "resolved_seed": request["resolved_seed"],
        "initial_scene_state_sha256": request["initial_scene_state_sha256"],
        "initial_measured_joint_state_sha256": request[
            "initial_measured_joint_state_sha256"
        ],
        "initial_commanded_drive_target_sha256": request[
            "initial_commanded_drive_target_sha256"
        ],
    }
    if dict(identity) != expected_identity:
        raise ConditionRunnerError("reset identity differs from frozen pair")
    root_query = backend.query(root_observation, 0)
    ordered, legal, fallback = _query_contract(root_query)
    registry_sha = canonical_sha256(
        {
            "pair_id": request["pair_id"], "candidate_count": CANDIDATE_COUNT,
            "ordered_candidate_sha256": ordered, "candidate_legal": legal,
        }
    )
    selector_proof: Mapping[str, Any]
    if request["condition"] == "baseline":
        selected = fallback
        selector_proof = {
            "selector": "lowest_legal_feasibility_root_candidate",
            "event_model_members_called": 0,
            "selected_candidate_index": selected,
            "score_contract": "lowest_legal_feasibility_root_candidate",
            "source_rank_score_contract_sha256": [],
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "formal190_target_outcome_calibrated_acceptance_margin": False,
        }
    else:
        selected, selector_proof = backend.select_etsf(root_query)
        if (
            type(selected) is not int or not 0 <= selected < CANDIDATE_COUNT
            or legal[selected] is not True
            or selector_proof.get("event_model_members_called") != 5
            or type(selector_proof.get("event_model_members_called")) is not int
            or selector_proof.get("uncertainty_gate_applied") is not True
            or selector_proof.get("score_contract")
            != "five_member_adjusted_source_composite_candidate_rank_score_margin"
            or not isinstance(
                selector_proof.get("source_rank_score_contract_sha256"), list
            )
            or len(selector_proof["source_rank_score_contract_sha256"]) != 5
            or any(
                not is_sha(value)
                for value in selector_proof["source_rank_score_contract_sha256"]
            )
            or selector_proof.get(
                "source_contract_rank_score_is_success_logit"
            ) is not False
            or selector_proof.get(
                "source_contract_rank_score_is_success_probability"
            ) is not False
            or selector_proof.get(
                "formal190_target_outcome_calibrated_acceptance_margin"
            ) is not True
            or selector_proof.get("source_rank_numeric_contract")
            != SOURCE_RANK_NUMERIC_CONTRACT
        ):
            raise ConditionRunnerError("ETSF selector did not prove five-member guarded inference")
    observation = root_observation
    query = root_query
    success = False
    steps: list[dict[str, Any]] = []
    terminated = truncated = False
    for step_index in range(int(backend.max_steps)):
        current_ordered, current_legal, current_lowest = _query_contract(query)
        chosen = selected if step_index == 0 else current_lowest
        if current_legal[chosen] is not True:
            raise ConditionRunnerError("selected candidate is not legal")
        mapped = np.asarray(query["mapped_actions"], dtype=np.float32)
        command = np.ascontiguousarray(mapped[chosen, 0])
        observation, terminated, truncated, info = backend.step(
            command.reshape(1, 1, ACTION_DIM)
        )
        if type(terminated) is not bool or type(truncated) is not bool \
           or type(info.get("success")) is not bool:
            raise ConditionRunnerError("simulator terminal/success values must be exact bool")
        success = success or info["success"]
        steps.append(
            {
                "step_index": step_index,
                "selection_role": (
                    "condition_selected_root" if step_index == 0
                    else "lowest_legal_continuation"
                ),
                "selected_candidate_index": chosen,
                "ordered_candidate_sha256": current_ordered,
                "candidate_legal": current_legal,
                "snapshot_sha256": canonical_sha256(dict(backend.snapshot())),
                "simulator_success": info["success"],
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        if terminated or truncated or step_index + 1 >= int(backend.max_steps):
            break
        query = backend.query(observation, step_index + 1)
    continuation_base = {
        "continuation_contract": CONTINUATION_CONTRACT,
        "continuation_policy_sha256": backend.continuation_policy_sha256,
        "continuation_rerank_after_root": False,
        "candidate_replacement_count": 0,
    }
    trajectory = {
        "pair_id": request["pair_id"], "condition": request["condition"],
        "attempt": 0, "root_candidate_registry_sha256": registry_sha,
        "selector_proof": dict(selector_proof), "steps": steps,
        "terminal_task_success": success,
        "terminal_task_success_source": "simulator_info_success_exact_bool",
        "predicted_success_used_as_outcome": False,
    }
    continuation = {
        **continuation_base,
        "root_selected_candidate_index": selected,
        "continuation_step_count": max(0, len(steps) - 1),
        "all_postroot_steps_lowest_legal": True,
    }
    result = {
        "pair_id": request["pair_id"], "ordinal": request["ordinal"],
        "attempt": 0, "condition": request["condition"],
        "condition_ordinal": request["condition_ordinal"],
        "shared_snapshot_sha256": request["shared_snapshot_sha256"],
        "candidate_count": CANDIDATE_COUNT,
        "ordered_candidate_sha256": ordered, "candidate_legal": legal,
        "candidate_registry_sha256": registry_sha,
        "schema6_execution_authority_file_sha256": (
            backend.schema6_execution_authority_file_sha256
        ),
        "schema6_runtime_contract_sha256": backend.schema6_runtime_contract_sha256,
        "max_episode_steps": 200,
        "selected_candidate_index": selected,
        "selector_execution_proof": dict(selector_proof),
        "selector_execution_proof_sha256": canonical_sha256(selector_proof),
        "selector_score_contract": selector_proof["score_contract"],
        "source_rank_score_contract_sha256": list(
            selector_proof["source_rank_score_contract_sha256"]
        ),
        "source_contract_rank_score_is_success_logit": selector_proof[
            "source_contract_rank_score_is_success_logit"
        ],
        "source_contract_rank_score_is_success_probability": selector_proof[
            "source_contract_rank_score_is_success_probability"
        ],
        "formal190_target_outcome_calibrated_acceptance_margin": selector_proof[
            "formal190_target_outcome_calibrated_acceptance_margin"
        ],
        **continuation_base,
        "continuation_proof_sha256": canonical_sha256(continuation_base),
        "task_success": success, "simulator_exit_code": 0,
    }
    return (
        result,
        json.dumps(trajectory, sort_keys=True, allow_nan=False).encode("utf-8"),
        json.dumps(continuation, sort_keys=True, allow_nan=False).encode("utf-8"),
    )


class ActualBackend:
    """Schema6 v2 runtime + dense four-candidate generation + five adapters."""

    def __init__(
        self, *, request: Mapping[str, Any], runtime_adapter: Any, dense: Any,
        trainer: Any, core: Mapping[str, Any], event_spec: Mapping[str, Any],
        execution_authority_path: Path, output_parent: Path,
        expected_runtime_contract_sha256: str,
    ) -> None:
        os.environ["ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY"] = str(execution_authority_path)
        contract = runtime_adapter._load_authority_runtime_contract()
        if (
            contract.get("runtime_contract_sha256")
            != expected_runtime_contract_sha256
            or type(contract.get("max_episode_steps")) is not int
            or contract["max_episode_steps"] != 200
        ):
            raise ConditionRunnerError("loaded simulator runtime is not exact full horizon")
        resources = runtime_adapter._build_resources(
            contract, output_parent=output_parent, load_policy=True
        )
        self._resources = resources
        self._runtime = resources["runtime"]
        self._dense = dense
        self._trainer = trainer
        self._request = request
        self._contract = contract
        self._event_spec = event_spec
        self._execution_authority_path = execution_authority_path
        self.max_steps = int(contract["max_episode_steps"])
        self.schema6_execution_authority_file_sha256 = file_sha256(
            execution_authority_path
        )
        self.schema6_runtime_contract_sha256 = str(
            contract["runtime_contract_sha256"]
        )
        self.continuation_policy_sha256 = canonical_sha256(
            {
                "contract": CONTINUATION_CONTRACT,
                "dense_collector_sha256": file_sha256(Path(str(dense.__file__))),
                "max_steps": self.max_steps,
            }
        )
        self._models = self._load_models(core)
        self._calibration, self._threshold = self._load_calibration(core)
        self._selector_authority, self._selector = self._load_selector_authority(core)
        self._root_snapshot: Mapping[str, Any] | None = None

    def _load_models(self, core: Mapping[str, Any]) -> list[tuple[Any, Any, Any, Any]]:
        import torch
        from openvla_etsf_event_world_model import (
            ActionConditionedEventWorldModel, EventWorldModelConfig,
        )

        members = core["r7h_target_adapter_lineage"]["members"]
        if not isinstance(members, list) or len(members) != 5:
            raise ConditionRunnerError("paired core does not bind five adapter members")
        models = []
        self._adapter_checkpoint_sha256: list[str] = []
        self._source_checkpoint_sha256: list[str] = []
        self._source_rank_contract_sha256: list[str] = []
        self._source_rank_success_temperatures: list[float] = []
        self._object_normalization_sha256: list[str] = []
        for index, member in enumerate(members):
            source_path = bound_file(
                Path(member["source_checkpoint"]["path"]),
                member["source_checkpoint"]["file_sha256"], f"source member {index}",
            )
            adapter_path = bound_file(
                Path(member["adapter_checkpoint"]["path"]),
                member["adapter_checkpoint"]["file_sha256"], f"adapter member {index}",
            )
            source = self._trainer._load_torch(source_path, f"source member {index}")
            source_file_sha = file_sha256(source_path)
            audit = self._trainer.validate_source_checkpoint(source)
            payload = self._trainer._load_torch(adapter_path, f"adapter member {index}")
            source_rank_contract = self._trainer._validate_source_rank_score_contract(
                payload.get("source_rank_score_contract")
            )
            adapter_config = payload.get("adapter_config")
            ranking_contract = payload.get("ranking_contract")
            if (
                source_file_sha != member["source_checkpoint"]["file_sha256"]
                or payload.get("source_checkpoint_sha256") != source_file_sha
                or source_rank_contract.get("source_checkpoint_file_sha256")
                != source_file_sha
                or not isinstance(adapter_config, Mapping)
                or adapter_config.get("source_action_rank_residual_consumed") is not True
                or adapter_config.get("source_action_rank_success_only") is not False
                or adapter_config.get("deployment_success_logit")
                != "base_factual_success_logit"
                or adapter_config.get("dense_success_uses_base_logit") is not True
                or adapter_config.get("deployment_primary_candidate_score")
                != "source_contract_rank_score"
                or adapter_config.get("source_contract_rank_score_is_success_logit")
                is not False
                or adapter_config.get(
                    "source_contract_rank_score_is_success_probability"
                ) is not False
                or adapter_config.get("source_rank_score_contract_sha256")
                != source_rank_contract["contract_sha256"]
                or not isinstance(ranking_contract, Mapping)
                or ranking_contract.get("candidate_prediction_api")
                != "predict_grouped_candidates"
                or ranking_contract.get("source_action_rank_success_only") is not False
                or ranking_contract.get("deployment_success_logit")
                != "base_factual_success_logit"
                or ranking_contract.get("deployment_primary_candidate_score")
                != "source_contract_rank_score"
                or ranking_contract.get("source_contract_rank_score_is_success_logit")
                is not False
                or ranking_contract.get(
                    "source_contract_rank_score_is_success_probability"
                ) is not False
                or ranking_contract.get(
                    "deployment_success_probability_selector_authorized"
                ) is not False
            ):
                raise ConditionRunnerError(
                    "source/adapter/composite-rank lineage contract changed"
                )
            config = EventWorldModelConfig.from_dict(audit["config"])
            object_mean, object_std = self._trainer.object_normalization(
                source, config.object_delta_dim
            )
            normalization_sha = self._trainer.array_bundle_sha256({
                "object_delta_mean": object_mean.cpu().numpy(),
                "object_delta_std": object_std.cpu().numpy(),
            })
            native = ActionConditionedEventWorldModel(config)
            native.load_state_dict(source["model"], strict=True)
            cfg = payload["adapter_config"]
            model = self._trainer.SmolVLAPiperAdapter(
                native,
                state_rank=int(cfg["state_rank"]),
                action_rank=int(cfg["action_rank"]),
                source_rank_contract=source_rank_contract,
            ).to(self._resources["device"])
            model.load_state_dict(payload["model"], strict=True)
            model.enforce_and_verify_frozen_core()
            model.eval()
            recovery_contract = payload.get("conditional_recovery_contract")
            if (
                not isinstance(recovery_contract, Mapping)
                or recovery_contract.get("trained") is not True
                or recovery_contract.get("shared_transition_stop_gradient") is not True
            ):
                raise ConditionRunnerError(
                    "all five conditional recovery heads must be trained and detached"
                )
            recovery = self._trainer.DetachedConditionalRecoveryAdapter(
                config.semantic_dim
            ).to(self._resources["device"])
            recovery.load_state_dict(
                payload["conditional_recovery_adapter"], strict=True
            )
            recovery.eval()
            models.append((
                model,
                recovery,
                object_mean.to(self._resources["device"]),
                object_std.to(self._resources["device"]),
            ))
            self._adapter_checkpoint_sha256.append(file_sha256(adapter_path))
            self._source_checkpoint_sha256.append(source_file_sha)
            self._source_rank_contract_sha256.append(
                source_rank_contract["contract_sha256"]
            )
            self._source_rank_success_temperatures.append(
                float(source_rank_contract["success_temperature"])
            )
            self._object_normalization_sha256.append(normalization_sha)
        return models

    def _load_selector_authority(
        self, core: Mapping[str, Any]
    ) -> tuple[dict[str, Any], Any]:
        authority = core.get("deployment", {}).get("selector_authority")
        fields = {
            "format", "status", "implementation", "utility_contract",
            "uncertainty_contract", "runtime_execution_authority_sha256",
            "deployment_uncertainty_implementation",
            "five_member_checkpoint_sha256", "calibration_sha256",
            "source_rank_score_contract_sha256",
            "source_rank_score_contracts", "deployment_parameters",
            "formal190_thresholds", "source_rank_numeric_contract",
            "source_rank_member_authority",
            "source_rank_member_authority_sha256",
            "object_source_normalization_sha256",
            "formal190_root_group_ranker_sha256",
            "selector_authority_sha256",
        }
        if not isinstance(authority, Mapping) or set(authority) != fields:
            raise ConditionRunnerError(
                "production six-head selector authority is not frozen in paired core"
            )
        logical = _verify_logical(
            authority, "selector_authority_sha256", "selector authority"
        )
        implementation = authority.get("implementation")
        utility = authority.get("utility_contract")
        uncertainty = authority.get("uncertainty_contract")
        uncertainty_implementation = authority.get(
            "deployment_uncertainty_implementation"
        )
        member_authority = authority.get("source_rank_member_authority")
        authority_members = (
            member_authority.get("members")
            if isinstance(member_authority, Mapping) else None
        )
        if (
            not isinstance(implementation, Mapping)
            or set(implementation) != {"path", "file_sha256"}
            or not isinstance(utility, Mapping)
            or utility.get("primary_score")
            != "five_member_adjusted_source_composite_candidate_rank_score_margin"
            or utility.get("primary_score_is_success_logit") is not False
            or utility.get("primary_score_is_success_probability") is not False
            or utility.get("scene_relative_candidate_comparison") is not True
            or utility.get("source_action_rank_residual_required") is not True
            or utility.get("source_action_rank_success_only") is not False
            or utility.get("piper_embodiment_adapter_required") is not True
            or utility.get("formal190_target_outcome_calibrated_acceptance_margin")
            is not True
            or utility.get("structured_heads_enter_primary_utility") is not False
            or utility.get("structured_heads_enter_uncertainty_and_ablation") is not True
            or not isinstance(uncertainty, Mapping)
            or not isinstance(uncertainty_implementation, Mapping)
            or set(uncertainty_implementation) != {"path", "file_sha256"}
            or not is_sha(uncertainty_implementation.get("file_sha256"))
            or uncertainty.get("calibration_scale_exact") is not True
            or uncertainty.get("aleatoric_and_epistemic_guard_only") is not True
            or uncertainty.get("object_predictions_physical_xyz_before_selector")
            is not True
            or uncertainty.get("duration_deployment_scale_applied_before_selector")
            is not True
            or uncertainty.get("object_deployment_scale_applied_before_selector")
            is not True
            or isinstance(
                uncertainty.get("formal190_object_error_robust_scale_m"), bool
            )
            or not isinstance(
                uncertainty.get("formal190_object_error_robust_scale_m"),
                (int, float),
            )
            or not math.isfinite(
                float(uncertainty["formal190_object_error_robust_scale_m"])
            )
            or float(uncertainty["formal190_object_error_robust_scale_m"]) <= 0.0
            or not is_sha(authority.get("runtime_execution_authority_sha256"))
            or not isinstance(authority.get("five_member_checkpoint_sha256"), list)
            or len(authority["five_member_checkpoint_sha256"]) != 5
            or any(not is_sha(value) for value in authority["five_member_checkpoint_sha256"])
            or not isinstance(authority.get("source_rank_score_contract_sha256"), list)
            or len(authority["source_rank_score_contract_sha256"]) != 5
            or any(
                not is_sha(value)
                for value in authority["source_rank_score_contract_sha256"]
            )
            or not isinstance(authority.get("source_rank_score_contracts"), list)
            or len(authority["source_rank_score_contracts"]) != 5
            or not isinstance(authority.get("deployment_parameters"), Mapping)
            or not isinstance(authority.get("formal190_thresholds"), Mapping)
            or authority.get("source_rank_numeric_contract")
            != SOURCE_RANK_NUMERIC_CONTRACT
            or not isinstance(member_authority, Mapping)
            or set(member_authority)
            != {"source_rank_numeric_contract", "members"}
            or member_authority.get("source_rank_numeric_contract")
            != SOURCE_RANK_NUMERIC_CONTRACT
            or not isinstance(authority_members, list)
            or len(authority_members) != 5
            or authority.get("source_rank_member_authority_sha256")
            != canonical_sha256(member_authority)
            or any(
                not isinstance(row, Mapping)
                or set(row)
                != {
                    "member_index", "source_checkpoint_file_sha256",
                    "source_rank_score_contract_sha256", "success_temperature",
                }
                or row.get("member_index") != index
                or row.get("source_checkpoint_file_sha256")
                != self._source_checkpoint_sha256[index]
                or row.get("source_rank_score_contract_sha256")
                != self._source_rank_contract_sha256[index]
                or isinstance(row.get("success_temperature"), bool)
                or not isinstance(row.get("success_temperature"), (int, float))
                or float(row["success_temperature"])
                != self._source_rank_success_temperatures[index]
                for index, row in enumerate(authority_members or [])
            )
            or not isinstance(authority.get("object_source_normalization_sha256"), list)
            or len(authority["object_source_normalization_sha256"]) != 5
            or any(
                not is_sha(value)
                for value in authority["object_source_normalization_sha256"]
            )
            or not is_sha(authority.get("calibration_sha256"))
            or not is_sha(authority.get("formal190_root_group_ranker_sha256"))
            or not is_sha(logical)
        ):
            raise ConditionRunnerError("six-head selector authority contract changed")
        if (
            authority["five_member_checkpoint_sha256"]
            != self._adapter_checkpoint_sha256
            or authority["source_rank_score_contract_sha256"]
            != self._source_rank_contract_sha256
            or authority["object_source_normalization_sha256"]
            != self._object_normalization_sha256
            or len(set(self._object_normalization_sha256)) != 1
            or self._object_normalization_sha256[0]
            != self._calibration.get("prediction_contract", {}).get(
                "object_source_normalization_sha256"
            )
            or authority["runtime_execution_authority_sha256"]
            != file_sha256(self._execution_authority_path)
            or authority["calibration_sha256"]
            != self._calibration.get("calibration_sha256")
            or authority["formal190_root_group_ranker_sha256"]
            != self._calibration.get("root_group_ranker", {}).get(
                "root_group_ranker_sha256"
            )
            or float(uncertainty["formal190_object_error_robust_scale_m"])
            != self._calibration.get("metrics", {}).get(
                "object_total_variance", {}
            ).get("deployment_object_error_robust_scale_m")
        ):
            raise ConditionRunnerError(
                "selector authority does not bind loaded members/calibration/runtime"
            )
        module = load_bound_module(
            Path(str(implementation["path"])),
            str(implementation["file_sha256"]),
            "six-head selector implementation",
        )
        if not callable(getattr(module, "select_root_candidate_v3", None)):
            raise ConditionRunnerError("selector implementation lacks reviewed v3 API")
        uncertainty_module = load_bound_module(
            Path(str(uncertainty_implementation["path"])),
            str(uncertainty_implementation["file_sha256"]),
            "deployment uncertainty implementation",
        )
        if not callable(getattr(uncertainty_module, "root_components", None)):
            raise ConditionRunnerError(
                "deployment uncertainty implementation lacks reviewed API"
            )
        return dict(authority), module

    @staticmethod
    def _load_calibration(core: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        bridge_record = core["evaluation400"]["identity_bridge"]
        bridge = read_json(
            Path(bridge_record["path"]), bridge_record["file_sha256"], "identity bridge"
        )
        dependency = bridge["dependencies"]["calibration"]
        calibration = read_json(
            Path(dependency["path"]), dependency["file_sha256"], "calibration"
        )
        logical = _verify_logical(calibration, "calibration_sha256", "calibration")
        if (
            set(dependency) != {"path", "file_sha256", "logical_sha256"}
            or logical != dependency["logical_sha256"]
        ):
            raise ConditionRunnerError("calibration dependency closure changed")
        threshold = calibration["abstain_threshold"]["maximum_total_uncertainty"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
           or not math.isfinite(float(threshold)) or float(threshold) < 0:
            raise ConditionRunnerError("calibration uncertainty threshold changed")
        if any(calibration["head_enabled_for_primary"].get(name) is not True for name in (
            "post_event", "next_event", "duration", "success", "object_effect", "recovery"
        )):
            raise ConditionRunnerError("all six calibrated heads are required")
        ranker = calibration.get("root_group_ranker")
        if (
            not isinstance(ranker, Mapping)
            or ranker.get("enabled_for_primary") is not True
            or ranker.get("score_semantics")
            != "source_contract_rank_score_minus_same_group_lowest_legal_baseline_then_five_member_mean"
            or ranker.get("score_is_success_logit") is not False
            or ranker.get("score_is_success_probability") is not False
            or not is_sha(ranker.get("root_group_ranker_sha256"))
            or _verify_logical(
                ranker, "root_group_ranker_sha256", "formal190 root ranker"
            ) != ranker["root_group_ranker_sha256"]
        ):
            raise ConditionRunnerError("formal190 composite root ranker is not enabled")
        duration_multiplier = calibration.get("metrics", {}).get(
            "duration_lognormal_mixture", {}
        ).get("deployment_scale_multiplier")
        object_multiplier = calibration.get("metrics", {}).get(
            "object_total_variance", {}
        ).get("deployment_scale_multiplier")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (duration_multiplier, object_multiplier)
        ):
            raise ConditionRunnerError("deployment uncertainty scales are invalid")
        return calibration, float(threshold)

    def reset(self, requested_seed: int) -> tuple[Any, Mapping[str, Any]]:
        from resolve_smolvla_piper_target_reset_only import array_sha256, scene_sha256
        from run_smolvla_piper_r6d_direct_actor_smoke import INSTRUCTION

        observation, resolved, observed = self._runtime.reset(requested_seed, INSTRUCTION)
        if observed != INSTRUCTION:
            raise ConditionRunnerError("runtime instruction changed")
        identity = self._runtime.identity_snapshot()
        self._root_snapshot = self._runtime.snapshot()
        return observation, {
            "resolved_seed": int(resolved),
            "initial_scene_state_sha256": scene_sha256(identity["scene_state"]),
            "initial_measured_joint_state_sha256": array_sha256(
                identity["measured_joint_state"], role="measured joint state"
            ),
            "initial_commanded_drive_target_sha256": array_sha256(
                identity["commanded_drive_target"], role="commanded drive target"
            ),
        }

    def query(self, observation: Any, query_index: int) -> Mapping[str, Any]:
        query = self._dense.generate_candidate_query(
            policy=self._resources["policy"], preprocessor=self._resources["preprocessor"],
            postprocessor=self._resources["postprocessor"], capture=self._resources["capture"],
            observation=observation, bounds=self._contract["piper_action_bounds"],
            device=self._resources["device"], scene_seed=int(self._request["requested_seed"]),
            query_index=query_index,
        )
        self._dense.validate_candidate_query(query)
        return query

    def _root_semantics(self) -> tuple[int, np.ndarray]:
        if self._root_snapshot is None:
            raise ConditionRunnerError("root semantics requested before reset")
        poses = np.asarray(self._root_snapshot["object_poses"], dtype=np.float32)[None]
        names = list(self._root_snapshot["object_names"])
        calibration_by_task = self._event_spec.get("calibration")
        if not isinstance(calibration_by_task, Mapping) or len(calibration_by_task) != 1:
            raise ConditionRunnerError("canonical event spec task calibration is ambiguous")
        calibration = next(iter(calibration_by_task.values()))
        stored = np.asarray([0], dtype=np.int64)
        predicates = self._trainer.reconstruct_pose_predicates(
            poses, names, False, calibration, stored
        )[0]
        return 0, predicates

    def select_etsf(self, query: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
        import torch

        current_event, predicates = self._root_semantics()
        legal = np.asarray(query["feasibility_mask"], dtype=bool)
        legal_indices = np.flatnonzero(legal).astype(np.int64)
        if len(legal_indices) < 2:
            raise ConditionRunnerError(
                "composite Source rank requires at least two legal root candidates"
            )
        row_count = len(legal_indices)
        state = np.repeat(
            np.asarray(query["hidden"], dtype=np.float32)[None], row_count, axis=0
        )
        all_actions = np.asarray(query["mapped_actions"], dtype=np.float32)
        actions = all_actions[legal_indices]
        proprio = np.repeat(
            np.asarray(query["processed_state"], dtype=np.float32)[None],
            row_count,
            axis=0,
        )
        fallback = int(query["lowest_legal_original_candidate_index"])
        logical_group_id = f"evaluation400:{self._request['pair_id']}:root"
        batch = {
            "state": torch.as_tensor(state, device=self._resources["device"]),
            "actions": torch.as_tensor(actions, device=self._resources["device"]),
            "action_mask": torch.as_tensor(
                np.arange(actions.shape[1])[None] == 0,
                device=self._resources["device"],
            ).expand(row_count, -1),
            "proprio": torch.as_tensor(proprio, device=self._resources["device"]),
            "current_event_id": torch.full(
                (row_count,), current_event, dtype=torch.long,
                device=self._resources["device"],
            ),
            "current_predicates": torch.as_tensor(
                np.repeat(predicates[None], row_count, axis=0),
                device=self._resources["device"],
            ),
            "history_mask": torch.ones(
                (row_count, 1), dtype=torch.bool, device=self._resources["device"]
            ),
            "dt": torch.ones(row_count, device=self._resources["device"]),
            "ranking_group_index": torch.zeros(
                row_count, dtype=torch.long, device=self._resources["device"]
            ),
            "ranking_candidate_index": torch.as_tensor(
                legal_indices, dtype=torch.long, device=self._resources["device"]
            ),
            "ranking_baseline_mask": torch.as_tensor(
                legal_indices == fallback,
                dtype=torch.bool,
                device=self._resources["device"],
            ),
            "ranking_group_count": 1,
            "ranking_logical_group_id": [logical_group_id] * row_count,
            "logical_group_id": [logical_group_id] * row_count,
        }
        outputs: list[tuple[Mapping[str, Any], Any]] = []
        duration_scale_multiplier = float(
            self._calibration["metrics"]["duration_lognormal_mixture"][
                "deployment_scale_multiplier"
            ]
        )
        object_scale_multiplier = float(
            self._calibration["metrics"]["object_total_variance"][
                "deployment_scale_multiplier"
            ]
        )
        with torch.no_grad():
            for model, recovery, object_mean, object_std in self._models:
                output = dict(model.predict_grouped_candidates(batch))
                deployment = apply_deployment_prediction_scales(
                    output,
                    object_mean=object_mean,
                    object_std=object_std,
                    duration_scale_multiplier=duration_scale_multiplier,
                    object_scale_multiplier=object_scale_multiplier,
                )
                output.update(deployment)
                outputs.append((output, recovery(output["transition"])))
        predictions = {
            "post_event_logits": np.stack(
                [row[0]["next_event_logits"].cpu().numpy() for row in outputs]
            ),
            "next_event_logits": np.stack(
                [row[0]["next_reached_event_logits"].cpu().numpy() for row in outputs]
            ),
            "duration_log_mean": np.stack(
                [row[0]["duration_log_mean"].cpu().numpy() for row in outputs]
            ),
            "duration_log_scale": np.stack(
                [row[0]["duration_log_scale"].cpu().numpy() for row in outputs]
            ),
            "success_logit": np.stack(
                [row[0]["success_logit"].cpu().numpy() for row in outputs]
            ),
            "source_contract_rank_score": np.stack(
                [
                    row[0]["source_contract_rank_score"].float().cpu().numpy()
                    for row in outputs
                ]
            ),
            "source_contract_base_rank_score": np.stack(
                [
                    row[0]["source_contract_base_rank_score"].float().cpu().numpy()
                    for row in outputs
                ]
            ),
            "source_action_rank_residual": np.stack(
                [
                    row[0]["action_rank_residual"].float().cpu().numpy()
                    for row in outputs
                ]
            ),
            "recovery_logit": np.stack([row[1].cpu().numpy() for row in outputs]),
            "object_mean": np.stack(
                [row[0]["object_mean"].cpu().numpy() for row in outputs]
            ),
            "object_log_scale": np.stack(
                [row[0]["object_log_scale"].cpu().numpy() for row in outputs]
            ),
        }
        raw = self._selector.select_root_candidate_v3(
            predictions=predictions,
            prediction_candidate_indices=legal_indices.copy(),
            candidate_legal=legal.copy(),
            fallback_index=fallback,
            calibration=dict(self._calibration),
            selector_authority=dict(self._selector_authority),
        )
        expected = {
            "selected_candidate_index", "proposed_candidate_index",
            "fallback_candidate_index", "score_margin", "total_uncertainty",
            "proposed_uncertainty", "baseline_uncertainty",
            "minimum_formal190_composite_margin",
            "maximum_formal190_pair_uncertainty",
            "maximum_global_total_uncertainty", "candidate_change_accepted",
            "decision_reason",
            "uncertainty_gate_applied", "five_member_call_count",
            "prediction_heads_computed", "primary_score_contract",
            "source_rank_score_contract_sha256",
            "source_rank_numeric_contract",
            "source_contract_rank_score_is_success_logit",
            "source_contract_rank_score_is_success_probability",
            "formal190_target_outcome_calibrated_acceptance_margin",
            "calibration_sha256", "formal190_root_group_ranker_sha256",
            "decision_algebra_sha256",
            "selector_input_sha256", "selector_input",
            "prediction_candidate_indices", "alternative_candidate_indices",
            "member_source_contract_rank_scores", "uncertainty_components",
            "member_source_contract_base_rank_scores",
            "member_source_action_rank_residuals",
            "member_source_rank_success_temperatures",
            "root_recovery_uncertainty_policy",
            "root_structured_uncertainty_head_count",
            "alternative_set_contract", "margin_comparison",
            "selector_proof_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ConditionRunnerError("reviewed selector output contract changed")
        proof_logical = _verify_logical(raw, "selector_proof_sha256", "selector proof")
        selected = raw["selected_candidate_index"]
        proposed = raw["proposed_candidate_index"]
        if (
            type(selected) is not int or not 0 <= selected < CANDIDATE_COUNT
            or legal[selected] is not True
            or type(proposed) is not int or not 0 <= proposed < CANDIDATE_COUNT
            or raw.get("fallback_candidate_index") != fallback
            or raw.get("uncertainty_gate_applied") is not True
            or type(raw.get("five_member_call_count")) is not int
            or raw["five_member_call_count"] != 5
            or raw.get("prediction_heads_computed")
            != [
                "post_event", "next_event", "duration", "success",
                "object_effect", "recovery", "source_contract_rank_score",
            ]
            or raw.get("primary_score_contract")
            != "five_member_adjusted_source_composite_candidate_rank_score_margin"
            or raw.get("source_rank_score_contract_sha256")
            != self._source_rank_contract_sha256
            or raw.get("source_rank_numeric_contract")
            != SOURCE_RANK_NUMERIC_CONTRACT
            or raw.get("source_contract_rank_score_is_success_logit") is not False
            or raw.get("source_contract_rank_score_is_success_probability") is not False
            or raw.get("formal190_target_outcome_calibrated_acceptance_margin")
            is not True
            or raw.get("calibration_sha256")
            != self._calibration["calibration_sha256"]
            or raw.get("formal190_root_group_ranker_sha256")
            != self._selector_authority["formal190_root_group_ranker_sha256"]
            or not is_sha(raw.get("decision_algebra_sha256"))
            or not isinstance(raw.get("selector_input"), Mapping)
            or not is_sha(raw.get("selector_input_sha256"))
            or raw["selector_input_sha256"]
            != canonical_sha256(raw["selector_input"])
            or raw.get("prediction_candidate_indices")
            != legal_indices.astype(int).tolist()
            or raw.get("alternative_candidate_indices")
            != [int(value) for value in legal_indices if int(value) != fallback]
            or raw.get("root_recovery_uncertainty_policy")
            != "excluded_at_initial_e0_without_observed_operational_regress"
            or type(raw.get("root_structured_uncertainty_head_count")) is not int
            or raw["root_structured_uncertainty_head_count"] != 5
            or raw.get("alternative_set_contract")
            != "all_legal_candidates_except_lowest_legal_baseline"
            or raw.get("margin_comparison")
            != "strict_greater_than_formal190_threshold"
            or not isinstance(raw.get("member_source_contract_rank_scores"), list)
            or not isinstance(raw.get("uncertainty_components"), Mapping)
            or any(
                isinstance(raw.get(name), bool)
                or not isinstance(raw.get(name), (int, float))
                or not math.isfinite(float(raw[name]))
                or float(raw[name]) < 0.0
                for name in (
                    "score_margin", "total_uncertainty", "proposed_uncertainty",
                    "baseline_uncertainty", "minimum_formal190_composite_margin",
                    "maximum_formal190_pair_uncertainty",
                    "maximum_global_total_uncertainty",
                )
            )
            or type(raw.get("candidate_change_accepted")) is not bool
            or raw["candidate_change_accepted"] is not (selected != fallback)
            or not is_sha(proof_logical)
        ):
            raise ConditionRunnerError("reviewed selector violated frozen authority")
        proof = {
            "selector": "frozen_five_member_event_world_model_with_uncertainty_abstention",
            "event_model_members_called": 5,
            "uncertainty_gate_applied": True,
            "selector_output_sha256": proof_logical,
            "selector_decision": dict(raw),
            "selected_candidate_index": selected,
            "proposed_candidate_index": proposed,
            "score_margin": float(raw["score_margin"]),
            "total_uncertainty": float(raw["total_uncertainty"]),
            "decision_algebra_sha256": raw["decision_algebra_sha256"],
            "calibration_sha256": raw["calibration_sha256"],
            "formal190_root_group_ranker_sha256": raw[
                "formal190_root_group_ranker_sha256"
            ],
            "score_contract": (
                "five_member_adjusted_source_composite_candidate_rank_score_margin"
            ),
            "source_rank_score_contract_sha256": list(
                self._source_rank_contract_sha256
            ),
            "source_rank_numeric_contract": SOURCE_RANK_NUMERIC_CONTRACT,
            "source_contract_rank_score_is_success_logit": False,
            "source_contract_rank_score_is_success_probability": False,
            "formal190_target_outcome_calibrated_acceptance_margin": True,
            "predicted_success_used_as_outcome": False,
        }
        proof["selector_proof_sha256"] = canonical_sha256(proof)
        return selected, proof

    def step(self, action: np.ndarray) -> tuple[Any, bool, bool, Mapping[str, Any]]:
        return self._runtime.step(action)

    def snapshot(self) -> Mapping[str, Any]:
        snapshot = self._runtime.snapshot()
        return {
            "object_names": list(snapshot["object_names"]),
            "object_poses_sha256": hashlib.sha256(np.asarray(snapshot["object_poses"]).tobytes()).hexdigest(),
            "proprio_sha256": hashlib.sha256(np.asarray(snapshot["proprio"]).tobytes()).hexdigest(),
            "control_step": int(snapshot["telemetry"]["control_step"]),
        }

    def close(self) -> None:
        self._resources["close"]()


def build_actual_backend(args: argparse.Namespace, request: Mapping[str, Any]) -> ActualBackend:
    paired_v3 = load_bound_module(
        args.paired_protocol_implementation,
        args.paired_protocol_implementation_file_sha256,
        "paired v3 protocol",
    )
    runtime = load_bound_module(
        args.simulator_implementation,
        args.simulator_implementation_file_sha256,
        "schema6 runtime adapter",
    )
    dense = load_bound_module(
        args.dense_collector_implementation,
        args.dense_collector_implementation_file_sha256,
        "dense collector",
    )
    trainer = load_bound_module(
        args.adapter_trainer_implementation,
        args.adapter_trainer_implementation_file_sha256,
        "adapter trainer",
    )
    core = read_json(args.protocol_core, args.protocol_core_file_sha256, "paired core")
    if paired_v3.validate_core(core) != core.get("protocol_core_sha256"):
        raise ConditionRunnerError("paired core logical SHA changed")
    decision = read_json(
        args.ed25519_decision, args.ed25519_decision_file_sha256, "execution decision"
    )
    bundle = read_json(
        args.execution_bundle, args.execution_bundle_file_sha256, "execution bundle"
    )
    if (
        paired_v3.verify_decision(
            decision, core=core, core_file_sha256=args.protocol_core_file_sha256
        ) != decision.get("decision_sha256")
        or paired_v3.validate_bundle(bundle) != bundle.get("bundle_sha256")
        or request["bundle_sha256"] != bundle["bundle_sha256"]
    ):
        raise ConditionRunnerError("core/decision/bundle closure changed")
    event_spec = read_json(
        args.canonical_event_spec,
        args.canonical_event_spec_file_sha256,
        "canonical event spec",
    )
    authority = bound_file(
        args.schema6_execution_authority,
        args.schema6_execution_authority_file_sha256,
        "schema6 execution authority",
    )
    runtime_contract = validate_full_horizon_runtime_binding(args, runtime)
    return ActualBackend(
        request=request, runtime_adapter=runtime, dense=dense, trainer=trainer,
        core=core, event_spec=event_spec, execution_authority_path=authority,
        output_parent=args.output_root.parent,
        expected_runtime_contract_sha256=runtime_contract[
            "runtime_contract_sha256"
        ],
    )


def validate_full_horizon_runtime_binding(
    args: argparse.Namespace, runtime_adapter: Any,
) -> dict[str, Any]:
    interface = read_json(
        args.runtime_contract, args.runtime_contract_file_sha256,
        "condition runner runtime interface",
    )
    interface_logical = _verify_logical(
        interface, "runtime_contract_sha256", "condition runner runtime interface"
    )
    expected_fields = {
        "format", "status", "interface_version", "mode", "request_format",
        "result_format", "condition_runner_implementation_sha256",
        "simulator_implementation_sha256", "visible_device_contract",
        "pair_attempt", "candidate_count", "condition_names",
        "schema6_execution_authority_file_sha256",
        "schema6_runtime_contract_sha256", "max_episode_steps",
        "runtime_contract_sha256",
    }
    if (
        set(interface) != expected_fields
        or interface.get("format")
        != "etsf_smolvla_piper_condition_runner_runtime_contract_v3"
        or interface.get("status")
        != "externally_reviewed_condition_runner_interface"
        or interface.get("interface_version") != 3
        or type(interface.get("interface_version")) is not int
        or interface.get("mode") != "execute-condition-v3"
        or interface.get("request_format") != REQUEST_FORMAT
        or interface.get("result_format") != RESULT_FORMAT
        or interface.get("condition_runner_implementation_sha256")
        != args.condition_runner_source_file_sha256
        or interface.get("simulator_implementation_sha256")
        != args.simulator_implementation_file_sha256
        or interface.get("visible_device_contract")
        != "exact_gpu_uuid_as_cuda_visible_devices_and_cuda0"
        or type(interface.get("pair_attempt")) is not int
        or interface["pair_attempt"] != 0
        or type(interface.get("candidate_count")) is not int
        or interface["candidate_count"] != CANDIDATE_COUNT
        or interface.get("condition_names") != ["baseline", "etsf"]
        or type(interface.get("max_episode_steps")) is not int
        or interface["max_episode_steps"] != 200
        or not is_sha(interface.get("schema6_execution_authority_file_sha256"))
        or not is_sha(interface.get("schema6_runtime_contract_sha256"))
        or not is_sha(interface_logical)
    ):
        raise ConditionRunnerError("condition runtime interface changed")
    authority_path = bound_file(
        args.schema6_execution_authority,
        args.schema6_execution_authority_file_sha256,
        "schema6 execution authority",
    )
    authority = read_json(
        authority_path, args.schema6_execution_authority_file_sha256,
        "schema6 execution authority",
    )
    if (
        interface["schema6_execution_authority_file_sha256"]
        != args.schema6_execution_authority_file_sha256
        or not isinstance(authority.get("runtime_contract"), Mapping)
    ):
        raise ConditionRunnerError("runtime interface authority binding changed")
    try:
        contract = runtime_adapter.validate_runtime_contract(
            authority["runtime_contract"]
        )
    except Exception as error:
        raise ConditionRunnerError("schema6 runtime contract is invalid") from error
    if (
        contract.get("runtime_contract_sha256")
        != interface["schema6_runtime_contract_sha256"]
        or type(contract.get("max_episode_steps")) is not int
        or contract["max_episode_steps"] != 200
    ):
        raise ConditionRunnerError("schema6 runtime authority is not exact 200 steps")
    return dict(contract)


def _write_new(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            if count <= 0:
                raise OSError("short result write")
            remaining = remaining[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def publish_result(
    *, output_root: Path, request_path: Path, request: Mapping[str, Any],
    result: Mapping[str, Any], trajectory: bytes, continuation: bytes,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.mkdir(mode=0o700)
    trajectory_path = output_root / "trajectory.bin"
    continuation_path = output_root / "continuation.bin"
    _write_new(trajectory_path, trajectory)
    _write_new(continuation_path, continuation)
    base = {
        "format": RESULT_FORMAT, "status": RESULT_STATUS,
        "request": {
            "path": str(request_path), "file_sha256": file_sha256(request_path),
            "logical_sha256": request["request_sha256"],
        },
        **dict(result),
        "trajectory_artifact": {
            "path": str(trajectory_path), "file_sha256": file_sha256(trajectory_path)
        },
        "continuation_artifact": {
            "path": str(continuation_path), "file_sha256": file_sha256(continuation_path)
        },
    }
    value = {**base, "result_sha256": canonical_sha256(base)}
    _write_new(
        output_root / "condition_result.json",
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n",
    )
    _write_new(output_root / "run.exit", b"0\n")
    output_root.chmod(0o555)
    return value


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    runtime = load_bound_module(
        args.simulator_implementation,
        args.simulator_implementation_file_sha256,
        "schema6 runtime adapter",
    )
    dense = load_bound_module(
        args.dense_collector_implementation,
        args.dense_collector_implementation_file_sha256,
        "dense collector",
    )
    trainer = load_bound_module(
        args.adapter_trainer_implementation,
        args.adapter_trainer_implementation_file_sha256,
        "adapter trainer",
    )
    runtime_api = {
        "validate_runtime_contract", "_load_authority_runtime_contract",
        "_build_resources",
    }
    dense_api = {"generate_candidate_query", "validate_candidate_query"}
    trainer_api = {
        "_load_torch", "validate_source_checkpoint",
        "_validate_source_rank_score_contract", "object_normalization",
        "array_bundle_sha256", "SmolVLAPiperAdapter",
        "DetachedConditionalRecoveryAdapter", "reconstruct_pose_predicates",
    }
    if (
        not all(callable(getattr(runtime, name, None)) for name in runtime_api)
        or not all(callable(getattr(dense, name, None)) for name in dense_api)
        or not all(callable(getattr(trainer, name, None)) for name in trainer_api)
    ):
        raise ConditionRunnerError("real condition execution API is incomplete")
    runtime_contract = validate_full_horizon_runtime_binding(args, runtime)
    return {
        "status": "preflight_only_no_simulator_or_policy_execution",
        "simulator_steps": 0, "policy_forwards": 0,
        "runtime_adapter_sha256": args.simulator_implementation_file_sha256,
        "dense_collector_sha256": args.dense_collector_implementation_file_sha256,
        "adapter_trainer_sha256": args.adapter_trainer_implementation_file_sha256,
        "schema6_runtime_contract_sha256": runtime_contract[
            "runtime_contract_sha256"
        ],
        "max_episode_steps": 200,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("execute-condition-v3", "preflight"), required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-file-sha256")
    parser.add_argument("--output-root", type=Path)
    for role in (
        "runtime-contract", "simulator-implementation", "protocol-core",
        "ed25519-decision", "execution-bundle", "canonical-event-spec",
        "schema6-execution-authority", "dense-collector-implementation",
        "adapter-trainer-implementation", "condition-runner-source",
        "paired-protocol-implementation", "local-dependency-closure",
    ):
        parser.add_argument(f"--{role}", type=Path, required=True)
        parser.add_argument(f"--{role}-file-sha256", required=True)
    parser.add_argument("--local-dependency-closure-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device != "cuda:0":
        raise ConditionRunnerError("condition runner requires isolated cuda:0")
    raw_core = read_json(
        args.protocol_core, args.protocol_core_file_sha256, "paired core"
    )
    selector = raw_core.get("deployment", {}).get("selector_authority")
    selector_implementation = (
        selector.get("implementation") if isinstance(selector, Mapping) else None
    )
    if (
        not isinstance(selector_implementation, Mapping)
        or set(selector_implementation) != {"path", "file_sha256"}
        or not is_sha(selector_implementation.get("file_sha256"))
    ):
        raise ConditionRunnerError(
            "production selector implementation is absent from paired core"
        )
    install_local_dependency_closure(
        path=args.local_dependency_closure,
        file_sha256_expected=args.local_dependency_closure_file_sha256,
        logical_sha256_expected=args.local_dependency_closure_sha256,
        expected_roots=[
            args.condition_runner_source,
            args.paired_protocol_implementation,
            args.simulator_implementation,
            args.dense_collector_implementation,
            args.adapter_trainer_implementation,
            Path(str(selector_implementation["path"])),
        ],
    )
    if args.mode == "preflight":
        print(json.dumps(preflight(args), sort_keys=True))
        return 0
    if args.request is None or args.request_file_sha256 is None or args.output_root is None:
        raise ConditionRunnerError("execute mode requires request and output root")
    request_path = bound_file(args.request, args.request_file_sha256, "condition request")
    request = validate_request(read_json(request_path, args.request_file_sha256, "condition request"))
    backend = build_actual_backend(args, request)
    try:
        result, trajectory, continuation = execute_condition(
            request=request, backend=backend
        )
        publish_result(
            output_root=args.output_root, request_path=request_path, request=request,
            result=result, trajectory=trajectory, continuation=continuation,
        )
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
