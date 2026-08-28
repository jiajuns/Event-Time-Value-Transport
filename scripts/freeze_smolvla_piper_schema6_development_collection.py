#!/usr/bin/env python3
"""Freeze independent authority for one non-Fresh Piper schema-v6 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePath
from typing import Any, Callable, Mapping

from etsf_schema6_pose_quality import registry_sha256, spec_sha256, validate_spec
from materialize_smolvla_piper_schema6_reset_contract import (
    build_pose_quality_spec,
    validate_materialized_registry_contract,
)
from execute_smolvla_piper_r6c_simulation_smoke import bind_r6c_preflight
from run_smolvla_piper_r6d_direct_actor_smoke import (
    atomic_json,
    bind_r6d_simulation_receipt,
    build_direct_actor_preregistration,
    canonical_sha256,
    file_sha256,
    reject_fresh_path,
    validate_direct_actor_preregistration,
)
from run_smolvla_piper_r6f_feasibility_smoke import (
    bind_r6e_preregistration,
    validate_feasibility_preregistration,
)


FORMAT = "smolvla_piper_schema6_development_collection_authority_v1"
STATUS = "frozen_development_collection_not_started"
TASK = "move_can_pot"
INSTRUCTION = "move the can into the pot"
FIXED_DEVELOPMENT_SEED = 100101000
MAX_ALLOWED_EPISODE_STEPS = 200
SENSITIVE_PATH_TOKENS = ("fresh", "confirmation")
HISTORICAL_IMPLEMENTATION_ROLES = (
    "direct_actor_runner",
    "r6d_base_executor",
    "shared_prefix_capture",
)


class CollectionAuthorityError(RuntimeError):
    """A frozen development collection authority is invalid or changed."""


def _sensitive_path_locations(value: Any) -> list[tuple[str, ...]]:
    """Find sensitive path strings lexically, without resolving any path."""

    result: list[tuple[str, ...]] = []

    def visit(item: Any, location: tuple[str, ...]) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, (*location, str(key)))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, (*location, f"[{index}]"))
        elif isinstance(item, str):
            looks_like_path = (
                item.startswith(("/", "./", "../"))
                or "\\" in item
                or ("/" in item and any(token in item.casefold() for token in SENSITIVE_PATH_TOKENS))
            )
            if looks_like_path and any(
                token in component.casefold()
                for component in PurePath(item).parts
                for token in SENSITIVE_PATH_TOKENS
            ):
                result.append(location)

    visit(value, ())
    return result


def _project_signed_development_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    """Replace the legacy path string with non-dereferenceable commitments."""

    if set(seed) != {
        "path", "sha256", "seed_registry", "requested_seed",
        "expected_resolved_seed", "fresh_confirmation_eligible", "label_free",
    }:
        raise CollectionAuthorityError("signed development seed fields changed")
    raw_path = seed.get("path")
    digest = seed.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path.startswith("/")
        or _sensitive_path_locations({"path": raw_path}) != [("path",)]
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or seed.get("seed_registry") != "explicit_v7_prospective_development"
        or seed.get("requested_seed") != FIXED_DEVELOPMENT_SEED
        or seed.get("expected_resolved_seed") != FIXED_DEVELOPMENT_SEED
        or seed.get("fresh_confirmation_eligible") is not False
        or seed.get("label_free") is not True
    ):
        raise CollectionAuthorityError("signed development seed metadata changed")
    projection: dict[str, Any] = {
        "format": "etsf_signed_legacy_seed_lineage_projection_v1",
        "seed_registry": seed["seed_registry"],
        "requested_seed": seed["requested_seed"],
        "expected_resolved_seed": seed["expected_resolved_seed"],
        "fresh_confirmation_eligible": False,
        "label_free": True,
        "legacy_path_value_sha256": hashlib.sha256(raw_path.encode("utf-8")).hexdigest(),
        "legacy_manifest_content_sha256": digest,
        "legacy_path_resolved": False,
        "legacy_path_stated": False,
        "legacy_path_opened": False,
        "legacy_path_dereferenced": False,
    }
    projection["projection_sha256"] = canonical_sha256(projection)
    return projection


def _historical_implementation_artifact(
    record: Mapping[str, Any], *, role: str, expected_sha256: str
) -> dict[str, str]:
    """Authenticate one old code path without substituting the current path."""

    if set(record) != {"path", "sha256"}:
        raise CollectionAuthorityError(f"historical R6e implementation fields changed: {role}")
    raw_text = record.get("path")
    if not isinstance(raw_text, str) or not raw_text.startswith("/"):
        raise CollectionAuthorityError(f"historical R6e implementation path is invalid: {role}")
    raw = Path(raw_text)
    if _sensitive_path_locations({"path": raw_text}) or raw.is_symlink():
        raise CollectionAuthorityError(f"historical R6e implementation path is unsafe: {role}")
    try:
        resolved = raw.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as exc:
        raise CollectionAuthorityError(
            f"historical R6e implementation is missing: {role}"
        ) from exc
    if (
        _sensitive_path_locations({"path": str(resolved)})
        or resolved != raw
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise CollectionAuthorityError(
            f"historical R6e implementation must be a canonical read-only regular file: {role}"
        )
    recorded_sha = record.get("sha256")
    if recorded_sha != expected_sha256 or file_sha256(resolved) != recorded_sha:
        raise CollectionAuthorityError(
            f"historical/current R6e implementation bytes differ: {role}"
        )
    return {"path": str(resolved), "sha256": str(recorded_sha)}


def _normalize_expected_r6e_to_historical_code_paths(
    signed_r6e: Mapping[str, Any], current_expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Permit only three byte-identical implementation path relocations.

    R6e was signed before schema6 code was copied into a versioned deployment
    root.  Rebuilding R6e with the current module's ``__file__`` changes three
    path strings and, consequently, its logical SHA.  All other fields must be
    exactly equal.  The historical files themselves remain required, frozen,
    regular, safe, and byte-identical to the current expected implementations.
    """

    historical_sources = signed_r6e.get("runtime_source_artifacts")
    expected_sources = current_expected.get("runtime_source_artifacts")
    if not isinstance(historical_sources, Mapping) or not isinstance(expected_sources, Mapping):
        raise CollectionAuthorityError("R6e runtime source artifacts are incomplete")
    normalized = json.loads(json.dumps(current_expected, sort_keys=True))
    for role in HISTORICAL_IMPLEMENTATION_ROLES:
        historical = historical_sources.get(role)
        current = expected_sources.get(role)
        if not isinstance(historical, Mapping) or not isinstance(current, Mapping):
            raise CollectionAuthorityError(f"R6e implementation role is missing: {role}")
        current_sha = current.get("sha256")
        if not isinstance(current_sha, str):
            raise CollectionAuthorityError(f"current R6e implementation SHA is invalid: {role}")
        normalized["runtime_source_artifacts"][role] = _historical_implementation_artifact(
            historical, role=role, expected_sha256=current_sha
        )
    base = {
        key: item for key, item in normalized.items() if key != "preregistration_sha256"
    }
    normalized["preregistration_sha256"] = canonical_sha256(base)
    return normalized


def load_r6f_lineage_for_collection(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Recompute R6f lineage even after its immutable output has been written.

    The canonical signed lineage embeds one old development-seed path whose
    namespace is now forbidden.  The path is never converted to ``Path``,
    resolved, stat'ed, or opened.  Its signed string and manifest digest are
    replaced in memory by a hash-only projection.  R6c/R6d and every safe
    runtime/model artifact remain fully rebound and rehashed.  No R6f receipt
    or outcome is read.
    """

    r6f = validate_feasibility_preregistration(path)
    if _sensitive_path_locations(r6f) != [
        ("inherited_R6e_contract", "development_seed", "path")
    ]:
        raise CollectionAuthorityError("R6f sensitive lineage locations changed")
    r6e_binding = bind_r6e_preregistration(Path(r6f["r6e_lineage"]["path"]))
    r6e = validate_direct_actor_preregistration(Path(r6e_binding["path"]))
    if _sensitive_path_locations(r6e) != [("development_seed", "path")]:
        raise CollectionAuthorityError("R6e sensitive lineage locations changed")
    if r6f["inherited_R6e_contract"]["development_seed"] != r6e["development_seed"]:
        raise CollectionAuthorityError("R6f/R6e signed development seed differs")
    r6c_record = r6e["r6c_binding"]
    r6d_record = r6e["r6d_binding"]
    r6c = bind_r6c_preflight(
        Path(r6c_record["manifest_path"]), Path(r6c_record["receipt_path"])
    )
    r6d = bind_r6d_simulation_receipt(
        Path(r6d_record["preregistration_path"]), Path(r6d_record["receipt_path"])
    )
    roots = {key: Path(value) for key, value in r6e["runtime_roots"].items()}
    expected_r6e = build_direct_actor_preregistration(
        r6c=r6c,
        r6d=r6d,
        seed=r6e["development_seed"],
        rlinf_root=roots["rlinf_root"],
        robotwin_root=roots["robotwin_root"],
        robotwin_code=roots["robotwin_code"],
        lerobot_root=roots["lerobot_root"],
        model_path=roots["model_path"],
        vlm_metadata_path=roots["vlm_metadata_path"],
        output=Path(r6e["output"]),
    )
    normalized_expected_r6e = _normalize_expected_r6e_to_historical_code_paths(
        r6e, expected_r6e
    )
    if r6e != normalized_expected_r6e:
        raise CollectionAuthorityError(
            "R6e differs beyond byte-identical historical implementation relocation"
        )
    inherited = {
        key: r6e[key]
        for key in (
            "r6c_binding", "r6d_binding", "development_seed", "runtime_roots",
            "runtime_source_artifacts", "vlm_metadata_bundle_sha256",
            "model_bundle_sha256", "capability_contract", "mapping_contract",
            "state_contract", "caveats",
        )
    }
    if r6f["r6e_lineage"] != r6e_binding:
        raise CollectionAuthorityError("R6f R6e-lineage binding changed")
    if r6f["inherited_R6e_contract"] != inherited or r6f[
        "inherited_R6e_contract_sha256"
    ] != canonical_sha256(inherited):
        raise CollectionAuthorityError("R6f inherited R6e contract changed")
    seed = _project_signed_development_seed(r6e["development_seed"])
    safe_r6e = json.loads(json.dumps(r6e, sort_keys=True))
    safe_r6e["development_seed"] = seed
    safe_r6f = json.loads(json.dumps(r6f, sort_keys=True))
    safe_r6f["inherited_R6e_contract"]["development_seed"] = seed
    if _sensitive_path_locations(safe_r6e) or _sensitive_path_locations(safe_r6f):
        raise CollectionAuthorityError("signed seed lineage projection retained a sensitive path")
    return safe_r6f, safe_r6e, r6c, r6d, seed


def _json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionAuthorityError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CollectionAuthorityError(f"{role} must contain an object")
    return value


def validate_event_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    calibration = value.get("calibration", {})
    chains = value.get("chains", {})
    if TASK not in calibration or TASK not in chains:
        raise CollectionAuthorityError("event spec lacks move_can_pot")
    chain = chains[TASK]
    if chain.get("merge_e1_e2") is not True or tuple(chain.get("chain", ())) != (
        "e0", "e12", "e3", "e4", "eK"
    ):
        raise CollectionAuthorityError("event spec canonical chain changed")
    task_calibration = calibration[TASK]
    if task_calibration.get("moving") != "can" or task_calibration.get("anchor") not in (None, "", "pot"):
        raise CollectionAuthorityError("event spec requires an object outside frozen can/pot registry")
    return json.loads(json.dumps(value, sort_keys=True))


def _artifact(path: Path, role: str) -> dict[str, str]:
    resolved = reject_fresh_path(path, role)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def build_collection_authority(
    *,
    r6f_preregistration: Mapping[str, Any],
    r6f_path: Path,
    object_registry_path: Path,
    pose_quality_spec_path: Path,
    event_spec_path: Path,
    output_directory: Path,
    max_episode_steps: int,
) -> dict[str, Any]:
    """Build a content-addressed authority distinct from the R6f smoke."""

    if type(max_episode_steps) is not int or not 1 <= max_episode_steps <= MAX_ALLOWED_EPISODE_STEPS:
        raise CollectionAuthorityError("max_episode_steps must be in [1,200]")
    output = reject_fresh_path(output_directory, "schema6 development output")
    if output.exists():
        raise FileExistsError(output)
    r6f_file = reject_fresh_path(r6f_path, "R6f preregistration")
    if r6f_preregistration.get("status") != "preregistered_R6f_feasibility_simulation_only_not_executed":
        raise CollectionAuthorityError("R6f lineage is not the frozen preregistration")
    if r6f_preregistration.get("explicit_instruction") != INSTRUCTION:
        raise CollectionAuthorityError("R6f instruction changed")
    inherited_seed = r6f_preregistration["inherited_R6e_contract"]["development_seed"]
    if (
        inherited_seed.get("requested_seed") != FIXED_DEVELOPMENT_SEED
        or inherited_seed.get("expected_resolved_seed") != FIXED_DEVELOPMENT_SEED
        or inherited_seed.get("fresh_confirmation_eligible") is not False
        or inherited_seed.get("label_free") is not True
    ):
        raise CollectionAuthorityError("R6f non-Fresh development seed contract changed")
    inherited_contract = r6f_preregistration["inherited_R6e_contract"]
    if inherited_contract.get("mapping_contract") != {
        "mode": "explicit_named_ordinal_angle_preserving_mapping",
        "derived_from_equal_14d_width": False,
        "kinematic_equivalence_claimed": False,
        "physical_equivalence_claimed": False,
        "clipping_or_scaling_forbidden": True,
    }:
        raise CollectionAuthorityError("explicit named Piper mapping contract changed")
    if inherited_contract.get("state_contract") != {
        "semantics": "[left drive_target q1..q6,left normalized gripper,right drive_target q1..q6,right normalized gripper]",
        "is_measured_qpos": False,
    }:
        raise CollectionAuthorityError("Piper drive-target state semantics changed")
    caveats = inherited_contract.get("caveats", {})
    if (
        caveats.get("reported_duration") != "policy row count not physical time"
        or caveats.get("performance_or_transfer_claim") is not False
    ):
        raise CollectionAuthorityError("duration/no-claim caveats changed")
    registry_artifact = _artifact(object_registry_path, "schema6 object registry")
    spec_artifact = _artifact(pose_quality_spec_path, "schema6 pose-quality spec")
    event_artifact = _artifact(event_spec_path, "event spec")
    task_source_path = Path(inherited_contract["runtime_roots"]["robotwin_code"]) / "envs/move_can_pot.py"
    task_source_artifact = _artifact(task_source_path, "move_can_pot runtime identity source")
    registry = validate_materialized_registry_contract(
        _json(Path(registry_artifact["path"]), "object registry")
    )
    registry_digest = registry_sha256(registry)
    spec = validate_spec(
        _json(Path(spec_artifact["path"]), "pose-quality spec"),
        expected_registry_sha256=registry_digest,
    )
    expected_spec = build_pose_quality_spec(
        registry, move_can_pot_source=task_source_artifact
    )
    if spec != expected_spec:
        raise CollectionAuthorityError(
            "pose-quality spec differs from the fixed reset-only materializer"
        )
    spec_digest = spec_sha256(spec, expected_registry_sha256=registry_digest)
    validate_event_spec(_json(Path(event_artifact["path"]), "event spec"))
    launcher = Path(__file__).with_name("launch_smolvla_piper_schema6_development_collection.py").resolve()
    collector = Path(__file__).with_name("collect_smolvla_piper_schema6_dense_event_branches.py").resolve()
    pose_quality = Path(__file__).with_name("etsf_schema6_pose_quality.py").resolve()
    materializer = Path(__file__).with_name("materialize_smolvla_piper_schema6_reset_contract.py").resolve()
    for source in (launcher, collector, pose_quality, materializer):
        if not source.is_file():
            raise FileNotFoundError(source)
    base_task_source = r6f_preregistration["inherited_R6e_contract"]["r6d_binding"][
        "runtime_source_artifacts"
    ]["robotwin_base_task"]
    base: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "authority_basis": {
            "user_authorized_complete_design_code_changes_and_training_start": True,
            "user_required_execution_only_on_designated_4090": True,
            "this_is_new_collection_authority": True,
            "R6f_four_step_smoke_reinterpreted_as_collection_authority": False,
        },
        "scope": {
            "evidence_scope": "nonfresh_piper_development_only",
            "task": TASK,
            "instruction": INSTRUCTION,
            "requested_seed": FIXED_DEVELOPMENT_SEED,
            "expected_resolved_seed": FIXED_DEVELOPMENT_SEED,
            "seed_count": 1,
            "candidate_indices": [0, 1, 2, 3],
            "root_minimum_legal_candidates": 2,
            "root_action_horizon": 1,
            "continuation_action_horizon": 1,
            "continuation_rule": "regenerate_four_then_lowest_legal_original_candidate_index",
            "max_episode_steps": max_episode_steps,
            "all_infeasible_behavior": "fail_closed_root_skip_or_continuation_right_censor",
            "tracked_pose_objects": ["can", "pot"],
        },
        "r6f_lineage": {
            "path": str(r6f_file),
            "file_sha256": file_sha256(r6f_file),
            "logical_sha256": r6f_preregistration["preregistration_sha256"],
            "authorization_role": "runtime_and_interface_lineage_not_collection_authority",
        },
        "inherited_runtime_contract_sha256": r6f_preregistration[
            "inherited_R6e_contract_sha256"
        ],
        "inherited_execution_semantics": {
            "mapping_contract": dict(
                inherited_contract["mapping_contract"]
            ),
            "state_contract": dict(
                inherited_contract["state_contract"]
            ),
            "caveats": dict(
                inherited_contract["caveats"]
            ),
        },
        "input_artifacts": {
            "object_registry": registry_artifact,
            "pose_quality_spec": spec_artifact,
            "event_spec": event_artifact,
        },
        "object_registry_sha256": registry_digest,
        "pose_integrity_spec_sha256": spec_digest,
        "object_identity_contract": {
            "move_can_pot_source": task_source_artifact,
            "stable_id_semantics": "task_attr_plus_live_SAPIEN_get_name_or_name",
            "asset_id_semantics": {
                "can": "task.can_id -> 105_sauce-can/baseN",
                "pot": "task.pot_id -> 060_kitchenpot/baseN",
            },
            "validated_on_every_reset_before_policy_forward_or_action": True,
            "random_clutter_table_wall_excluded": True,
        },
        "implementation_sources": {
            "freezer": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
            "launcher": {"path": str(launcher), "sha256": file_sha256(launcher)},
            "collector": {"path": str(collector), "sha256": file_sha256(collector)},
            "pose_quality": {"path": str(pose_quality), "sha256": file_sha256(pose_quality)},
            "registry_materializer": {"path": str(materializer), "sha256": file_sha256(materializer)},
        },
        "telemetry_source_binding": {
            "robotwin_base_task": dict(base_task_source),
            "counted_call_site": "BaseTask.gen_sparse_reward_data TOPP loop self.scene.step",
            "counter_mode": "per-reset_instance_bound_scene_step_wrapper",
            "timestamp_mode": "cumulative_scene_step_count_times_scene_get_timestep",
            "physics_substeps_mode": "counted_scene_step_calls_since_previous_snapshot",
        },
        "output_contract": {
            "directory": str(output),
            "must_be_absent_before_launch": True,
            "manifest": str(output / "manifest.json"),
            "receipt": str(output / "collection_receipt.json"),
            "group": str(output / f"group_seed_{FIXED_DEVELOPMENT_SEED}.hdf5"),
            "overwrite_authorized": False,
        },
        "capability_contract": {
            "simulation_collection_authorized": True,
            "real_robot_execution_authorized": False,
            "fresh_inputs_allowed": False,
            "fresh_trajectory_or_label_opened": False,
            "development_labels_may_be_recorded": True,
            "performance_evaluation_authorized": False,
            "task_success_claim_authorized": False,
            "transfer_claim_authorized": False,
        },
        "exit_contract": {
            "success": 0,
            "root_fewer_than_two_legal": 20,
            "contract_or_runtime_failure": 21,
            "receipt_written_for_every_post_output_creation_exit": True,
        },
    }
    return {**base, "authority_sha256": canonical_sha256(base)}


def validate_collection_authority(
    path: Path,
    *,
    r6f_loader: Callable[[Path], tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]] = load_r6f_lineage_for_collection,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    authority_path = reject_fresh_path(path, "schema6 collection authority")
    authority = _json(authority_path, "schema6 collection authority")
    recorded = authority.get("authority_sha256")
    base = {key: value for key, value in authority.items() if key != "authority_sha256"}
    if recorded != canonical_sha256(base):
        raise CollectionAuthorityError("collection authority logical SHA changed")
    if authority.get("format") != FORMAT or authority.get("status") != STATUS:
        raise CollectionAuthorityError("unexpected collection authority format/status")
    capability = authority.get("capability_contract")
    if capability != {
        "simulation_collection_authorized": True,
        "real_robot_execution_authorized": False,
        "fresh_inputs_allowed": False,
        "fresh_trajectory_or_label_opened": False,
        "development_labels_may_be_recorded": True,
        "performance_evaluation_authorized": False,
        "task_success_claim_authorized": False,
        "transfer_claim_authorized": False,
    }:
        raise CollectionAuthorityError("collection Fresh/no-claim capability changed")
    basis = authority.get("authority_basis", {})
    if basis.get("this_is_new_collection_authority") is not True or basis.get(
        "R6f_four_step_smoke_reinterpreted_as_collection_authority"
    ) is not False:
        raise CollectionAuthorityError("new-authority/R6f boundary changed")
    scope = authority.get("scope", {})
    if (
        scope.get("requested_seed") != FIXED_DEVELOPMENT_SEED
        or scope.get("expected_resolved_seed") != FIXED_DEVELOPMENT_SEED
        or scope.get("seed_count") != 1
        or scope.get("candidate_indices") != [0, 1, 2, 3]
        or scope.get("root_minimum_legal_candidates") != 2
        or scope.get("root_action_horizon") != 1
        or scope.get("continuation_action_horizon") != 1
        or scope.get("tracked_pose_objects") != ["can", "pot"]
        or not 1 <= int(scope.get("max_episode_steps", 0)) <= MAX_ALLOWED_EPISODE_STEPS
    ):
        raise CollectionAuthorityError("collection seed/candidate/H1/step scope changed")
    for role, artifact in authority.get("implementation_sources", {}).items():
        source = reject_fresh_path(Path(artifact["path"]), f"implementation {role}")
        if file_sha256(source) != artifact.get("sha256"):
            raise CollectionAuthorityError(f"implementation source changed: {role}")
    for role, artifact in authority.get("input_artifacts", {}).items():
        source = reject_fresh_path(Path(artifact["path"]), f"input {role}")
        if file_sha256(source) != artifact.get("sha256"):
            raise CollectionAuthorityError(f"input artifact changed: {role}")
    lineage = authority["r6f_lineage"]
    r6f_path = Path(lineage["path"])
    if file_sha256(r6f_path) != lineage["file_sha256"]:
        raise CollectionAuthorityError("R6f preregistration file changed")
    r6f, r6e, r6c, r6d, seed = r6f_loader(r6f_path)
    if r6f["preregistration_sha256"] != lineage["logical_sha256"]:
        raise CollectionAuthorityError("R6f logical lineage changed")
    expected = build_collection_authority(
        r6f_preregistration=r6f,
        r6f_path=r6f_path,
        object_registry_path=Path(authority["input_artifacts"]["object_registry"]["path"]),
        pose_quality_spec_path=Path(authority["input_artifacts"]["pose_quality_spec"]["path"]),
        event_spec_path=Path(authority["input_artifacts"]["event_spec"]["path"]),
        output_directory=Path(authority["output_contract"]["directory"]),
        max_episode_steps=int(scope["max_episode_steps"]),
    )
    if authority != expected:
        raise CollectionAuthorityError("collection authority differs from full recomputation")
    return authority, r6f, r6e, r6c, r6d, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6f-preregistration", type=Path, required=True)
    parser.add_argument("--object-registry", type=Path, required=True)
    parser.add_argument("--pose-quality-spec", type=Path, required=True)
    parser.add_argument("--event-spec", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-episode-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r6f, _, _, _, _ = load_r6f_lineage_for_collection(
        args.r6f_preregistration
    )
    authority = build_collection_authority(
        r6f_preregistration=r6f,
        r6f_path=args.r6f_preregistration,
        object_registry_path=args.object_registry,
        pose_quality_spec_path=args.pose_quality_spec,
        event_spec_path=args.event_spec,
        output_directory=args.output_directory,
        max_episode_steps=args.max_episode_steps,
    )
    authority_output = reject_fresh_path(args.output, "schema6 collection authority output")
    if authority_output.exists():
        raise FileExistsError(authority_output)
    atomic_json(authority_output, authority)
    authority_output.chmod(0o444)
    print(json.dumps({
        "status": STATUS,
        "path": str(authority_output),
        "file_sha256": file_sha256(authority_output),
        "authority_sha256": authority["authority_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
