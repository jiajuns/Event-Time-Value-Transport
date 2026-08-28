#!/usr/bin/env python3
"""Verify an R6f feasibility receipt without replaying policy or simulator.

The verifier authenticates the fully recomputed R6f preregistration, the exact
runner implementation, and every value that the receipt makes independently
recomputable.  In particular, it reconstructs float32 processed states, shared
960-D prefixes, raw drive targets, and all sixteen mapped first actions; it
then recomputes their array hashes and every Piper feasibility decision.

The current R6f receipt intentionally stores only SHA-256 commitments for the
full 50x14 postprocessed chunks, CUDA noise tensors, and RGB observations.  A
hash is not reversible, so this verifier checks those commitments' syntax and
cross-links but does not falsely call them independently recomputed.  The
returned evidence ceiling records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from run_smolvla_piper_r6d_direct_actor_smoke import (
    ACTION_DIM,
    ACTION_EXEC_STEPS,
    ACTOR_ID,
    CHUNK_SIZE,
    DIRECT_MAX_STEPS,
    EXPECTED_RESOLVED_DEVELOPMENT_SEED,
    INSTRUCTION,
    PREFIX_DIM,
    REQUESTED_DEVELOPMENT_SEED,
    _load_and_recompute_preregistration,
    canonical_sha256,
)
from run_smolvla_piper_r6f_feasibility_smoke import (
    CANDIDATE_COUNT,
    FORMAT,
    R6E_EXPECTED_UNSAFE_TARGET,
    R6E_EXPECTED_UNSAFE_TARGET_INDEX,
    R6E_EXPECTED_UNSAFE_VALUE,
    FeasibilitySmokeError,
    assess_candidate_first_action,
    bind_r6e_preregistration,
    explicit_named_map_first_action,
    validate_feasibility_preregistration,
)
from verify_smolvla_piper_zero_shot_preflight import (
    PIPER_ACTION_SLOTS,
    array_sha256,
    file_sha256,
    reject_fresh_path,
)


VERIFICATION_FORMAT = "smolvla_piper_r6f_feasibility_receipt_verification_v1"
COMPLETED_STATUS = "completed_R6f_feasibility_simulation_interface_smoke"
SELECTION_RULE = "lowest_candidate_index_with_finite_in_bounds_first_action"


class FeasibilityReceiptVerificationError(FeasibilitySmokeError):
    """An R6f receipt is unauthenticated, inconsistent, or insufficient."""


def load_completed_feasibility_preregistration(
    path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Recompute R6f lineage after its bound receipt has been materialized.

    The execution-time R6f loader deliberately reconstructs its authority by
    calling a builder that requires the output receipt to be absent.  Reusing
    that loader after a successful run therefore raises ``FileExistsError``.
    A verifier must not move, delete, or hide the authenticated receipt merely
    to satisfy an execution-time precondition.  This read-only path instead
    validates the frozen R6f document, recomputes its complete R6e/R6c/R6d/seed
    lineage, and proves that every inherited R6e section is still identical.
    """

    prereg = validate_feasibility_preregistration(path)
    recorded_r6e = prereg.get("r6e_lineage")
    if not isinstance(recorded_r6e, Mapping):
        raise FeasibilityReceiptVerificationError(
            "R6f preregistration lacks its R6e lineage"
        )
    r6e = bind_r6e_preregistration(Path(str(recorded_r6e.get("path", ""))))
    if dict(recorded_r6e) != r6e:
        raise FeasibilityReceiptVerificationError(
            "R6f preregistration R6e lineage differs from read-only recomputation"
        )
    r6e_prereg, r6c, r6d, seed = _load_and_recompute_preregistration(
        Path(r6e["path"])
    )
    inherited_keys = (
        "r6c_binding",
        "r6d_binding",
        "development_seed",
        "runtime_roots",
        "runtime_source_artifacts",
        "vlm_metadata_bundle_sha256",
        "model_bundle_sha256",
        "capability_contract",
        "mapping_contract",
        "state_contract",
        "caveats",
    )
    recomputed_inherited = {key: r6e_prereg[key] for key in inherited_keys}
    if prereg.get("inherited_R6e_contract") != recomputed_inherited:
        raise FeasibilityReceiptVerificationError(
            "R6f inherited R6e contract differs from read-only recomputation"
        )
    if prereg.get("inherited_R6e_contract_sha256") != canonical_sha256(
        recomputed_inherited
    ):
        raise FeasibilityReceiptVerificationError(
            "R6f inherited R6e contract logical SHA mismatch"
        )
    return prereg, r6e_prereg, r6c, r6d, seed


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FeasibilityReceiptVerificationError(
            f"{role} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise FeasibilityReceiptVerificationError(f"{role} must be an object")
    return value


def _exact_fields(value: Any, expected: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FeasibilityReceiptVerificationError(f"{role} must be an object")
    actual = set(value)
    if actual != expected:
        raise FeasibilityReceiptVerificationError(
            f"{role} fields changed: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(value: Any, role: str) -> str:
    if not _is_sha256(value):
        raise FeasibilityReceiptVerificationError(
            f"{role} must be one lowercase SHA-256 digest"
        )
    return value


def _plain_bool(value: Any, role: str) -> bool:
    if type(value) is not bool:
        raise FeasibilityReceiptVerificationError(f"{role} must be boolean")
    return value


def _plain_int(value: Any, role: str) -> int:
    if type(value) is not int:
        raise FeasibilityReceiptVerificationError(f"{role} must be an integer")
    return value


def _float32_vector(value: Any, length: int, role: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise FeasibilityReceiptVerificationError(
            f"{role} must contain exactly {length} values"
        )
    if any(type(item) not in (int, float) for item in value):
        raise FeasibilityReceiptVerificationError(
            f"{role} must contain only JSON numbers"
        )
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise FeasibilityReceiptVerificationError(
            f"{role} must be finite float32[{length}]"
        )
    return np.ascontiguousarray(result)


def _piper_bounds() -> list[list[float]]:
    return [[float(slot.lower), float(slot.upper)] for slot in PIPER_ACTION_SLOTS]


def _bound_r6c_bounds(r6c: Mapping[str, Any]) -> list[list[float]]:
    try:
        raw = r6c["receipt"]["static_contract"]["static_semantics"][
            "piper_action_bounds"
        ]
    except (KeyError, TypeError) as exc:
        raise FeasibilityReceiptVerificationError(
            "fully recomputed R6c binding lacks Piper bounds"
        ) from exc
    try:
        result = [[float(pair[0]), float(pair[1])] for pair in raw]
    except (TypeError, ValueError, IndexError) as exc:
        raise FeasibilityReceiptVerificationError(
            "fully recomputed R6c Piper bounds are malformed"
        ) from exc
    if result != _piper_bounds():
        raise FeasibilityReceiptVerificationError(
            "R6c Piper bounds differ from the named local slot registry"
        )
    return result


def _verify_file_record(
    record: Any, role: str, *, expected: Mapping[str, Any] | None = None
) -> dict[str, str]:
    item = _exact_fields(record, {"path", "sha256"}, role)
    path = reject_fresh_path(Path(str(item["path"])), role)
    if not path.is_file():
        raise FeasibilityReceiptVerificationError(f"{role} file is missing")
    digest = _sha256(item["sha256"], f"{role}.sha256")
    if file_sha256(path) != digest:
        raise FeasibilityReceiptVerificationError(f"{role} file SHA changed")
    if expected is not None:
        expected_path = reject_fresh_path(Path(str(expected.get("path", ""))), role)
        if path != expected_path or digest != expected.get("sha256"):
            raise FeasibilityReceiptVerificationError(
                f"{role} differs from preregistration"
            )
    return {"path": str(path), "sha256": digest}


def _verify_input_interface(value: Any, query_index: int) -> None:
    role = f"execution.queries[{query_index}].input_interface"
    interface = _exact_fields(
        value,
        {
            "drive_target",
            "state_shape",
            "state_sha256",
            "main_image_shape",
            "main_image_sha256",
            "wrist_images_shape",
            "wrist_images_sha256",
        },
        role,
    )
    drive_target = _float32_vector(
        interface["drive_target"], ACTION_DIM, f"{role}.drive_target"
    )
    if interface["state_shape"] != [1, ACTION_DIM]:
        raise FeasibilityReceiptVerificationError(
            f"{role}.state_shape must be [1,14]"
        )
    if interface["state_sha256"] != array_sha256(drive_target.reshape(1, -1)):
        raise FeasibilityReceiptVerificationError(
            f"{role}.state_sha256 does not match drive_target"
        )
    main_shape = interface["main_image_shape"]
    wrists_shape = interface["wrist_images_shape"]
    if (
        not isinstance(main_shape, list)
        or len(main_shape) != 4
        or main_shape[0] != 1
        or main_shape[-1] != 3
        or any(type(dim) is not int or dim <= 0 for dim in main_shape)
    ):
        raise FeasibilityReceiptVerificationError(
            f"{role}.main_image_shape is not [1,H,W,3]"
        )
    if (
        not isinstance(wrists_shape, list)
        or len(wrists_shape) != 5
        or wrists_shape[0] != 1
        or wrists_shape[1] != 2
        or wrists_shape[-1] != 3
        or any(type(dim) is not int or dim <= 0 for dim in wrists_shape)
    ):
        raise FeasibilityReceiptVerificationError(
            f"{role}.wrist_images_shape is not [1,2,H,W,3]"
        )
    _sha256(interface["main_image_sha256"], f"{role}.main_image_sha256")
    _sha256(interface["wrist_images_sha256"], f"{role}.wrist_images_sha256")


def _verify_candidate(
    value: Any,
    *,
    query_index: int,
    candidate_index: int,
    prefix_sha256: str,
    bounds: Sequence[Sequence[float]],
) -> tuple[dict[str, Any], str, str]:
    role = (
        f"execution.queries[{query_index}].candidate_records[{candidate_index}]"
    )
    candidate = _exact_fields(
        value,
        {
            "candidate_index",
            "noise_sha256",
            "prefix_sha256",
            "postprocessed_chunk_sha256",
            "mapped_first_action_sha256",
            "first_action",
            "feasibility",
        },
        role,
    )
    if candidate.get("candidate_index") != candidate_index:
        raise FeasibilityReceiptVerificationError(
            f"{role}.candidate_index is not positional"
        )
    noise_sha = _sha256(candidate.get("noise_sha256"), f"{role}.noise_sha256")
    chunk_sha = _sha256(
        candidate.get("postprocessed_chunk_sha256"),
        f"{role}.postprocessed_chunk_sha256",
    )
    if candidate.get("prefix_sha256") != prefix_sha256:
        raise FeasibilityReceiptVerificationError(
            f"{role}.prefix_sha256 differs from the recomputed shared prefix"
        )
    action = _float32_vector(
        candidate.get("first_action"), ACTION_DIM, f"{role}.first_action"
    )
    if candidate.get("mapped_first_action_sha256") != array_sha256(action):
        raise FeasibilityReceiptVerificationError(
            f"{role}.mapped_first_action_sha256 mismatch"
        )
    expected_feasibility = assess_candidate_first_action(
        action,
        bounds,
        query_index=query_index,
        candidate_index=candidate_index,
    )
    if candidate.get("feasibility") != expected_feasibility:
        raise FeasibilityReceiptVerificationError(
            f"{role}.feasibility differs from independent bounds recomputation"
        )
    return dict(candidate), noise_sha, chunk_sha


def _verify_query(
    value: Any,
    *,
    query_index: int,
    bounds: Sequence[Sequence[float]],
) -> tuple[dict[str, Any], set[str], set[str]]:
    role = f"execution.queries[{query_index}]"
    query = _exact_fields(
        value,
        {
            "query_index",
            "selection_rule",
            "selection_uses_event_or_utility_score",
            "processed_state",
            "processed_state_sha256",
            "shared_prefix",
            "candidate_prefix_sha256",
            "prefix_bit_exact_across_all_four_candidates",
            "candidate_records",
            "selected_candidate_index",
            "input_interface",
            "env_step_performed",
            "action_horizon_per_env_step",
        },
        role,
    )
    if query.get("query_index") != query_index:
        raise FeasibilityReceiptVerificationError(
            f"{role}.query_index is not positional"
        )
    if (
        query.get("selection_rule") != SELECTION_RULE
        or query.get("selection_uses_event_or_utility_score") is not False
    ):
        raise FeasibilityReceiptVerificationError(
            f"{role} selection contract changed"
        )
    state = _float32_vector(
        query.get("processed_state"), ACTION_DIM, f"{role}.processed_state"
    )
    if query.get("processed_state_sha256") != array_sha256(state):
        raise FeasibilityReceiptVerificationError(
            f"{role}.processed_state_sha256 mismatch"
        )
    prefix = _float32_vector(
        query.get("shared_prefix"), PREFIX_DIM, f"{role}.shared_prefix"
    )
    prefix_sha = array_sha256(prefix)
    prefix_hashes = query.get("candidate_prefix_sha256")
    if prefix_hashes != [prefix_sha] * CANDIDATE_COUNT:
        raise FeasibilityReceiptVerificationError(
            f"{role}.candidate_prefix_sha256 is not four copies of the "
            "recomputed 960-D prefix hash"
        )
    if query.get("prefix_bit_exact_across_all_four_candidates") is not True:
        raise FeasibilityReceiptVerificationError(
            f"{role} does not assert bit-exact candidate prefixes"
        )
    records = query.get("candidate_records")
    if not isinstance(records, list) or len(records) != CANDIDATE_COUNT:
        raise FeasibilityReceiptVerificationError(
            f"{role} must contain exactly four candidate records"
        )
    verified: list[dict[str, Any]] = []
    noise_hashes: set[str] = set()
    chunk_hashes: set[str] = set()
    for candidate_index, candidate in enumerate(records):
        record, noise_sha, chunk_sha = _verify_candidate(
            candidate,
            query_index=query_index,
            candidate_index=candidate_index,
            prefix_sha256=prefix_sha,
            bounds=bounds,
        )
        verified.append(record)
        if noise_sha in noise_hashes:
            raise FeasibilityReceiptVerificationError(
                f"{role} reuses a candidate noise hash"
            )
        noise_hashes.add(noise_sha)
        chunk_hashes.add(chunk_sha)
    accepted = [
        index
        for index, record in enumerate(verified)
        if record["feasibility"]["accepted"] is True
    ]
    if not accepted:
        raise FeasibilityReceiptVerificationError(
            f"{role} has no legal candidate but claims a full H=1 step"
        )
    if query.get("selected_candidate_index") != accepted[0]:
        raise FeasibilityReceiptVerificationError(
            f"{role} did not select the lowest legal candidate index"
        )
    if query.get("env_step_performed") is not True:
        raise FeasibilityReceiptVerificationError(
            f"{role} did not execute its selected legal candidate"
        )
    if query.get("action_horizon_per_env_step") != ACTION_EXEC_STEPS:
        raise FeasibilityReceiptVerificationError(
            f"{role} violated the frozen H=1 execution contract"
        )
    _verify_input_interface(query.get("input_interface"), query_index)
    return dict(query), noise_hashes, chunk_hashes


def _verify_candidate0_diagnostic(
    value: Any, first_candidate: Mapping[str, Any]
) -> None:
    role = "r6e_candidate0_diagnostic_independently_recomputed"
    diagnostic = _exact_fields(
        value,
        {
            "accepted",
            "first_action",
            "feasibility",
            "matches_reported_left_joint2_failure_approximately",
        },
        role,
    )
    if diagnostic.get("accepted") is not False:
        raise FeasibilityReceiptVerificationError(
            "R6e candidate0 diagnostic must remain rejected"
        )
    if (
        diagnostic.get("first_action") != first_candidate.get("first_action")
        or diagnostic.get("feasibility") != first_candidate.get("feasibility")
    ):
        raise FeasibilityReceiptVerificationError(
            "R6e candidate0 diagnostic differs from query0/candidate0"
        )
    violations = first_candidate.get("feasibility", {}).get("violations", [])
    matches = [
        item
        for item in violations
        if isinstance(item, Mapping)
        and item.get("target_index") == R6E_EXPECTED_UNSAFE_TARGET_INDEX
        and item.get("target_joint_name") == R6E_EXPECTED_UNSAFE_TARGET
        and item.get("reason") == "outside_bounds"
        and item.get("allowed") == [0.0, 3.14]
        and type(item.get("value")) in (int, float)
        and np.isclose(
            float(item["value"]),
            R6E_EXPECTED_UNSAFE_VALUE,
            rtol=0.0,
            atol=5e-7,
        )
    ]
    if len(matches) != 1:
        raise FeasibilityReceiptVerificationError(
            "query0/candidate0 does not reproduce the R6e left:joint2 failure"
        )
    if diagnostic.get("matches_reported_left_joint2_failure_approximately") is not True:
        raise FeasibilityReceiptVerificationError(
            "R6e candidate0 diagnostic match flag changed"
        )


def verify_feasibility_receipt(
    preregistration_path: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    preregistration_loader: Callable[
        [Path],
        tuple[
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ],
    ] = load_completed_feasibility_preregistration,
) -> dict[str, Any]:
    """Authenticate one complete 4-query x 4-candidate R6f receipt.

    ``expected_receipt_sha256`` is deliberately mandatory: a receipt cannot
    authenticate its own mutable file identity.
    """

    prereg_path = reject_fresh_path(preregistration_path, "R6f preregistration")
    receipt_file = reject_fresh_path(receipt_path, "R6f receipt")
    expected_receipt_sha = _sha256(
        expected_receipt_sha256, "expected receipt SHA256"
    )
    if not prereg_path.is_file() or not receipt_file.is_file():
        raise FeasibilityReceiptVerificationError(
            "R6f preregistration/receipt file is missing"
        )
    receipt_file_sha = file_sha256(receipt_file)
    if receipt_file_sha != expected_receipt_sha:
        raise FeasibilityReceiptVerificationError("R6f receipt file SHA mismatch")
    prereg, _r6e_prereg, r6c, r6d, seed = preregistration_loader(prereg_path)
    bounds = _bound_r6c_bounds(r6c)
    receipt = _load_json(receipt_file, "R6f receipt")
    top = _exact_fields(
        receipt,
        {
            "format",
            "status",
            "actor_id",
            "source_body",
            "target_body",
            "target_runtime",
            "real_robot_execution",
            "fresh_inputs_used",
            "fresh_trajectory_or_label_opened",
            "task_success_claimed",
            "performance_evaluation_authorized",
            "transfer_claim_authorized",
            "event_or_utility_scoring_performed",
            "preregistration",
            "r6e_lineage",
            "r6e_candidate0_diagnostic_independently_recomputed",
            "development_seed_contract",
            "environment_contract",
            "loaded_policy_contract",
            "execution",
            "time_contract",
            "implementation_sha256",
            "interpretation",
        },
        "R6f receipt",
    )
    if (
        top.get("format") != FORMAT
        or top.get("status") != COMPLETED_STATUS
        or top.get("actor_id") != ACTOR_ID
        or top.get("source_body") != "aloha"
        or top.get("target_body") != "piper"
        or top.get("target_runtime") != "RoboTwin_simulation_only"
    ):
        raise FeasibilityReceiptVerificationError(
            "R6f receipt identity/status/body/runtime changed"
        )
    for key in (
        "real_robot_execution",
        "fresh_inputs_used",
        "fresh_trajectory_or_label_opened",
        "task_success_claimed",
        "performance_evaluation_authorized",
        "transfer_claim_authorized",
        "event_or_utility_scoring_performed",
    ):
        if top.get(key) is not False:
            raise FeasibilityReceiptVerificationError(
                f"R6f Fresh/capability boundary changed at {key}"
            )
    if top.get("interpretation") != (
        "fixed-candidate feasibility fallback only; no ETSF/event ranking, "
        "task performance, transfer, safety, or real-robot claim"
    ):
        raise FeasibilityReceiptVerificationError(
            "R6f interpretation/evidence ceiling changed"
        )
    prereg_record = _exact_fields(
        top.get("preregistration"),
        {"path", "file_sha256", "logical_sha256"},
        "receipt.preregistration",
    )
    recorded_prereg_path = reject_fresh_path(
        Path(str(prereg_record["path"])), "receipt.preregistration.path"
    )
    if recorded_prereg_path != prereg_path:
        raise FeasibilityReceiptVerificationError(
            "receipt binds a different R6f preregistration path"
        )
    if prereg_record.get("file_sha256") != file_sha256(prereg_path):
        raise FeasibilityReceiptVerificationError(
            "receipt R6f preregistration file SHA mismatch"
        )
    if prereg_record.get("logical_sha256") != prereg.get(
        "preregistration_sha256"
    ):
        raise FeasibilityReceiptVerificationError(
            "receipt R6f preregistration logical SHA mismatch"
        )
    if reject_fresh_path(Path(str(prereg.get("output"))), "R6f output") != receipt_file:
        raise FeasibilityReceiptVerificationError(
            "receipt path differs from preregistered output"
        )
    if top.get("r6e_lineage") != prereg.get("r6e_lineage"):
        raise FeasibilityReceiptVerificationError(
            "receipt R6e lineage differs from preregistration"
        )
    if top.get("development_seed_contract") != seed:
        raise FeasibilityReceiptVerificationError(
            "receipt development seed differs from fully recomputed binding"
        )

    runner_record = _exact_fields(
        prereg.get("r6f_runner"), {"path", "sha256"}, "prereg.r6f_runner"
    )
    runner_path = reject_fresh_path(
        Path(str(runner_record["path"])), "R6f runner"
    )
    runner_sha = _sha256(runner_record.get("sha256"), "prereg.r6f_runner.sha256")
    if (
        not runner_path.is_file()
        or file_sha256(runner_path) != runner_sha
        or top.get("implementation_sha256") != runner_sha
    ):
        raise FeasibilityReceiptVerificationError(
            "R6f receipt does not bind the authenticated runner implementation"
        )

    environment = _exact_fields(
        top.get("environment_contract"),
        {
            "embodiment",
            "requested_seed",
            "resolved_seed",
            "explicit_instruction",
            "scene_seed_and_instruction_strictly_bound",
            "center_crop",
            "collect_wrist_camera",
            "state_is_measured_qpos",
            "runtime_module_origins",
            "eval_seed_registry",
        },
        "environment_contract",
    )
    if environment.get("embodiment") != ["piper", "piper", 0.6]:
        raise FeasibilityReceiptVerificationError("Piper embodiment changed")
    if (
        environment.get("requested_seed") != REQUESTED_DEVELOPMENT_SEED
        or environment.get("resolved_seed") != EXPECTED_RESOLVED_DEVELOPMENT_SEED
        or environment.get("explicit_instruction") != INSTRUCTION
        or environment.get("scene_seed_and_instruction_strictly_bound") is not False
        or environment.get("center_crop") is not False
        or environment.get("collect_wrist_camera") is not True
        or environment.get("state_is_measured_qpos") is not False
    ):
        raise FeasibilityReceiptVerificationError(
            "R6f environment/seed/instruction contract changed"
        )
    inherited_sources = r6d.get("runtime_source_artifacts")
    if not isinstance(inherited_sources, Mapping):
        raise FeasibilityReceiptVerificationError(
            "fully recomputed R6d binding lacks runtime module sources"
        )
    origins = environment.get("runtime_module_origins")
    expected_origin_roles = {
        "rlinf_robotwin_env",
        "robotwin_vector_env",
        "robotwin_base_task",
        "robotwin_robot_controller",
    }
    if not isinstance(origins, Mapping) or set(origins) != expected_origin_roles:
        raise FeasibilityReceiptVerificationError(
            "runtime module origin roles changed"
        )
    for role in sorted(expected_origin_roles):
        _verify_file_record(origins[role], f"runtime_module_origins.{role}", expected=inherited_sources[role])
    _verify_file_record(
        environment.get("eval_seed_registry"),
        "environment_contract.eval_seed_registry",
        expected=prereg["inherited_R6e_contract"]["runtime_source_artifacts"][
            "robotwin_eval_seed_registry"
        ],
    )

    policy = _exact_fields(
        top.get("loaded_policy_contract"),
        {
            "checkpoint_declared_state_dim",
            "runtime_preprocessed_state_shape",
            "checkpoint_action_dim",
            "runtime_postprocessed_action_shape",
            "state_dimension_conflict_retained",
            "normalizer_state_dim",
        },
        "loaded_policy_contract",
    )
    if (
        policy.get("checkpoint_declared_state_dim") != 6
        or policy.get("runtime_preprocessed_state_shape") != [1, ACTION_DIM]
        or policy.get("checkpoint_action_dim") != ACTION_DIM
        or policy.get("runtime_postprocessed_action_shape")
        not in ([CHUNK_SIZE, ACTION_DIM], [1, CHUNK_SIZE, ACTION_DIM])
        or policy.get("state_dimension_conflict_retained") is not True
        or policy.get("normalizer_state_dim") != ACTION_DIM
    ):
        raise FeasibilityReceiptVerificationError(
            "loaded SmolVLA 6/14/14 policy contract changed"
        )

    execution = _exact_fields(
        top.get("execution"),
        {
            "queries_performed",
            "steps_executed",
            "max_steps",
            "action_exec_steps",
            "candidate_count_per_query",
            "selection_rule",
            "feasibility_baseline_only",
            "event_or_utility_scoring_performed",
            "no_feasible_candidate_halt",
            "stopped_on_termination",
            "stopped_on_truncation",
            "success_observed_diagnostic_only",
            "all_env_actions_prevalidated",
            "silent_clipping_possible",
            "mapping_contract",
            "queries",
        },
        "execution",
    )
    if (
        execution.get("queries_performed") != DIRECT_MAX_STEPS
        or execution.get("steps_executed") != DIRECT_MAX_STEPS
        or execution.get("max_steps") != DIRECT_MAX_STEPS
        or execution.get("action_exec_steps") != ACTION_EXEC_STEPS
        or execution.get("candidate_count_per_query") != CANDIDATE_COUNT
        or execution.get("selection_rule") != SELECTION_RULE
        or execution.get("feasibility_baseline_only") is not True
        or execution.get("event_or_utility_scoring_performed") is not False
        or execution.get("no_feasible_candidate_halt") is not False
        or execution.get("stopped_on_termination") is not False
        # The frozen RoboTwin smoke has step_limit == max_steps == 4, so the
        # fourth successful H=1 step is expected to raise the time-limit
        # truncation flag.  This is completion, not an early-stop failure.
        or execution.get("stopped_on_truncation") is not True
        or execution.get("all_env_actions_prevalidated") is not True
        or execution.get("silent_clipping_possible") is not False
    ):
        raise FeasibilityReceiptVerificationError(
            "execution is not the complete time-limit-truncated "
            "4-query x 4-candidate H=1 feasibility smoke"
        )
    _plain_bool(
        execution.get("success_observed_diagnostic_only"),
        "execution.success_observed_diagnostic_only",
    )
    _, expected_mapping = explicit_named_map_first_action(
        np.zeros(ACTION_DIM, dtype=np.float32)
    )
    if execution.get("mapping_contract") != expected_mapping:
        raise FeasibilityReceiptVerificationError(
            "execution named action mapping contract changed"
        )
    queries = execution.get("queries")
    if not isinstance(queries, list) or len(queries) != DIRECT_MAX_STEPS:
        raise FeasibilityReceiptVerificationError(
            "execution must contain exactly four queries"
        )
    all_noise_hashes: set[str] = set()
    chunk_commitments = 0
    first_query: dict[str, Any] | None = None
    for query_index, query in enumerate(queries):
        verified_query, query_noise, query_chunks = _verify_query(
            query, query_index=query_index, bounds=bounds
        )
        if all_noise_hashes.intersection(query_noise):
            raise FeasibilityReceiptVerificationError(
                "candidate noise hash is reused across queries"
            )
        all_noise_hashes.update(query_noise)
        chunk_commitments += len(query_chunks)
        if query_index == 0:
            first_query = verified_query
    assert first_query is not None
    _verify_candidate0_diagnostic(
        top.get("r6e_candidate0_diagnostic_independently_recomputed"),
        first_query["candidate_records"][0],
    )
    if top.get("time_contract") != {
        "unit": "policy action row count",
        "physical_duration_claimed": False,
    }:
        raise FeasibilityReceiptVerificationError(
            "R6f time/physical-duration claim contract changed"
        )

    receipt_logical_sha = canonical_sha256(receipt)
    return {
        "format": VERIFICATION_FORMAT,
        "status": "passed_complete_R6f_receipt_internal_consistency",
        "receipt": {
            "path": str(receipt_file),
            "file_sha256": receipt_file_sha,
            "logical_sha256": receipt_logical_sha,
        },
        "preregistration": {
            "path": str(prereg_path),
            "file_sha256": file_sha256(prereg_path),
            "logical_sha256": prereg["preregistration_sha256"],
        },
        "runner": {"path": str(runner_path), "sha256": runner_sha},
        "verified_execution": {
            "queries": DIRECT_MAX_STEPS,
            "candidates_per_query": CANDIDATE_COUNT,
            "candidate_records": DIRECT_MAX_STEPS * CANDIDATE_COUNT,
            "steps_executed": DIRECT_MAX_STEPS,
            "action_horizon": ACTION_EXEC_STEPS,
            "lowest_legal_selection_recomputed": True,
            "candidate0_R6e_diagnostic_recomputed": True,
            "fresh_used": False,
            "performance_or_transfer_claim": False,
        },
        "hash_evidence": {
            "processed_state_arrays_recomputed": DIRECT_MAX_STEPS,
            "raw_drive_target_arrays_recomputed": DIRECT_MAX_STEPS,
            "shared_prefix_arrays_recomputed": DIRECT_MAX_STEPS,
            "candidate_prefix_hash_links_recomputed": DIRECT_MAX_STEPS
            * CANDIDATE_COUNT,
            "mapped_first_action_arrays_recomputed": DIRECT_MAX_STEPS
            * CANDIDATE_COUNT,
            "postprocessed_chunk_commitments_syntax_checked": chunk_commitments,
            "postprocessed_chunks_independently_recomputed": False,
            "noise_commitments_syntax_checked": len(all_noise_hashes),
            "noise_tensors_independently_recomputed": False,
            "RGB_commitments_syntax_checked": DIRECT_MAX_STEPS * 2,
            "RGB_arrays_independently_recomputed": False,
        },
        "evidence_ceiling": (
            "simulation-only feasibility receipt; full 50x14 chunks, CUDA noise "
            "tensors, and RGB arrays are absent from the receipt, so their hashes "
            "remain authenticated-format commitments rather than independently "
            "recomputed evidence; no performance, task-success, transfer, safety, "
            "ETSF-ranking, physical-equivalence, or real-robot claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    args = parser.parse_args()
    result = verify_feasibility_receipt(
        args.preregistration,
        args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
