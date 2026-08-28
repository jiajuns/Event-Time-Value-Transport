from __future__ import annotations

import copy
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_smolvla_piper_zero_shot_preflight import (  # noqa: E402
    ACTOR_ID,
    FORMAT,
    SOURCE_DATASET_FRAME_COUNT,
    PreflightError,
    StateDimensionConflict,
    adapt_aloha_source_actions_to_piper_forward_interface,
    array_sha256,
    expected_slot_mapping_contract,
    expected_state_dimension_resolution,
    file_sha256,
    reject_fresh_path,
    run_preflight,
    validate_candidate_actions,
    validate_shared_prefix,
    validate_static_preflight,
)
from freeze_smolvla_piper_zero_shot_preflight_manifest import (  # noqa: E402
    _write_pair,
    build_manifest,
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    dtype_names = {
        np.dtype("int64"): "I64",
        np.dtype("float32"): "F32",
    }
    header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        array = np.ascontiguousarray(value)
        raw = array.tobytes()
        header[name] = {
            "dtype": dtype_names[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


def _stats(*, include_state: bool) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for prefix in (["action", "observation.state"] if include_state else ["action"]):
        result[f"{prefix}.count"] = np.array([SOURCE_DATASET_FRAME_COUNT], dtype=np.int64)
        result[f"{prefix}.min"] = np.full(14, -1.0, dtype=np.float32)
        result[f"{prefix}.max"] = np.full(14, 1.0, dtype=np.float32)
        result[f"{prefix}.mean"] = np.zeros(14, dtype=np.float32)
        result[f"{prefix}.std"] = np.ones(14, dtype=np.float32)
    return result


def _urdf(path: Path, limits: dict[str, tuple[float, float]]) -> None:
    joints = "".join(
        f'<joint name="{name}" type="'
        f'{"prismatic" if name.endswith("joint7") else "revolute"}'
        f'"><limit lower="{lower}" '
        f'upper="{upper}" effort="1" velocity="1"/></joint>'
        for name, (lower, upper) in limits.items()
    )
    path.write_text(f"<robot name=\"fixture\">{joints}</robot>", encoding="utf-8")


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


@pytest.fixture()
def bundle(tmp_path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, str]:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = model_path / "config.json"
    weights_path = model_path / "model.safetensors"
    train_path = model_path / "train_config.json"
    pre_path = model_path / "policy_preprocessor.json"
    post_path = model_path / "policy_postprocessor.json"
    pre_stats_path = model_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    post_stats_path = model_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    piper_dir = tmp_path / "RoboTwin" / "assets" / "embodiments" / "piper"
    aloha_dir = tmp_path / "RoboTwin" / "assets" / "embodiments" / "aloha-agilex"
    (aloha_dir / "urdf").mkdir(parents=True)
    piper_dir.mkdir(parents=True)
    piper_config_path = piper_dir / "config.yml"
    aloha_config_path = aloha_dir / "config.yml"
    piper_urdf_path = piper_dir / "piper.urdf"
    aloha_urdf_path = aloha_dir / "urdf" / "arx5_description_isaac.urdf"

    feature_state_6 = {"type": "STATE", "shape": [6]}
    feature_action_14 = {"type": "ACTION", "shape": [14]}
    _json(
        config_path,
        {
            "repo_id": "pepijn223/smolvla_robotwin",
            "input_features": {"observation.state": feature_state_6},
            "output_features": {"action": feature_action_14},
            "normalization_mapping": {
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
            "use_delta_joint_actions_aloha": False,
        },
    )
    weights_path.write_bytes(b"fixture-model-weights")
    _json(
        train_path,
        {
            "dataset": {
                "repo_id": "pepijn223/robotwin_unified_v3",
                "episodes": None,
            },
            "policy": {
                "input_features": {"observation.state": feature_state_6},
                "output_features": {"action": feature_action_14},
            },
            "env": {
                "features": {
                    "agent_pos": {"type": "STATE", "shape": [14]},
                    "action": feature_action_14,
                }
            },
        },
    )
    _json(
        pre_path,
        {
            "steps": [
                {
                    "registry_name": "normalizer_processor",
                    "config": {
                        "features": {
                            "observation.state": feature_state_6,
                            "action": feature_action_14,
                        }
                    },
                    "state_file": pre_stats_path.name,
                }
            ]
        },
    )
    _json(
        post_path,
        {
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": {"action": feature_action_14}},
                    "state_file": post_stats_path.name,
                }
            ]
        },
    )
    _safetensors(pre_stats_path, _stats(include_state=True))
    _safetensors(post_stats_path, _stats(include_state=False))

    yaml.safe_dump(
        {
            "urdf_path": "./piper.urdf",
            "arm_joints_name": [[f"joint{i}" for i in range(1, 7)]] * 2,
            "gripper_name": [{"base": "joint7"}, {"base": "joint7"}],
            "gripper_scale": [-0.01, 0.04],
            "dual_arm": False,
        },
        piper_config_path.open("w", encoding="utf-8"),
    )
    yaml.safe_dump(
        {
            "urdf_path": "./urdf/arx5_description_isaac.urdf",
            "arm_joints_name": [
                [f"fl_joint{i}" for i in range(1, 7)],
                [f"fr_joint{i}" for i in range(1, 7)],
            ],
            "gripper_name": [{"base": "fl_joint7"}, {"base": "fr_joint7"}],
            "gripper_scale": [-0.01, 0.045],
            "dual_arm": True,
        },
        aloha_config_path.open("w", encoding="utf-8"),
    )
    piper_arm_limits = [
        (-2.618, 2.618),
        (0.0, 3.14),
        (-2.697, 2.697),
        (-1.832, 1.832),
        (-1.22, 1.22),
        (-3.14, 3.14),
    ]
    _urdf(
        piper_urdf_path,
        {**{f"joint{i+1}": pair for i, pair in enumerate(piper_arm_limits)}, "joint7": (0.0, 0.04)},
    )
    _urdf(
        aloha_urdf_path,
        {
            **{f"fl_joint{i}": (-10.0, 10.0) for i in range(1, 7)},
            "fl_joint7": (0.0, 0.04765),
            **{f"fr_joint{i}": (-10.0, 10.0) for i in range(1, 7)},
            "fr_joint7": (0.0, 0.04765),
        },
    )
    actions = np.zeros((4, 50, 14), dtype=np.float32)
    actions[:, :, [1, 8]] = 1.0
    actions[:, :, [6, 13]] = 0.5
    for candidate in range(4):
        actions[candidate, :, [0, 2, 3, 4, 5, 7, 9, 10, 11, 12]] += 0.01 * candidate
    prefixes = np.broadcast_to(
        np.arange(960, dtype=np.float32).reshape(1, 960), (4, 960)
    ).copy()
    actions_path = tmp_path / "aloha_source_candidates.npy"
    prefixes_path = tmp_path / "shared_prefixes.npy"
    image_path = tmp_path / "source.png"
    probe_path = tmp_path / "forward_probe.json"
    np.save(actions_path, actions, allow_pickle=False)
    np.save(prefixes_path, prefixes, allow_pickle=False)
    image_path.write_bytes(b"fixture-png")
    probe = {
        "schema_version": 1,
        "experiment_type": "interface_smoke_not_task_success",
        "model_path": str(model_path.resolve()),
        "image_source": [str(image_path.resolve())],
        "candidate_generator": "native_smolvla_flow_matching_explicit_noise",
        "preprocessing": "checkpoint_preprocessor_and_postprocessor",
        "model_config_observation_state_dim": 6,
        "runtime_observation_state_dim": 14,
        "runtime_preprocessed_observation_state_shape": [1, 14],
        "state_dimension_override_used": True,
        "candidate_count": 4,
        "candidate_shape": [4, 50, 14],
        "identical_candidate_pairs": 0,
        "native_multi_candidate_verified": True,
        "array_outputs": {
            "candidate_actions": str(actions_path.resolve()),
            "candidate_actions_array_sha256": array_sha256(actions),
            "shared_prefixes": str(prefixes_path.resolve()),
            "shared_prefix_array_sha256": array_sha256(prefixes[0]),
        },
        "etsf_shared_state_hook": {
            "shape": [4, 960],
            "feature_dim": 960,
            "hook_calls_per_candidate": [1, 1, 1, 1],
            "max_abs_delta_from_candidate_0": [0.0, 0.0, 0.0, 0.0],
            "bit_exact_across_noise_candidates": True,
            "candidate_specific_expert_hidden_saved": False,
            "status": "verified",
        },
        "task_success_claimed": False,
    }
    _json(probe_path, probe)
    manifest: dict[str, object] = {
        "format": FORMAT,
        "actor_id": ACTOR_ID,
        "source_body": "aloha",
        "target_body": "piper",
        "artifacts": {
            "checkpoint_config": _artifact(config_path),
            "model_weights": _artifact(weights_path),
            "train_config": _artifact(train_path),
            "policy_preprocessor": _artifact(pre_path),
            "policy_postprocessor": _artifact(post_path),
            "preprocessor_stats": _artifact(pre_stats_path),
            "postprocessor_stats": _artifact(post_stats_path),
            "piper_body_config": _artifact(piper_config_path),
            "aloha_body_config": _artifact(aloha_config_path),
            "piper_urdf": _artifact(piper_urdf_path),
            "aloha_urdf": _artifact(aloha_urdf_path),
        },
        "probe_artifacts": {
            "forward_probe_receipt": _artifact(probe_path),
            "candidate_actions": _artifact(actions_path),
            "shared_prefixes": _artifact(prefixes_path),
            "source_image": _artifact(image_path),
        },
        "slot_mapping_contract": expected_slot_mapping_contract(),
        "state_dimension_resolution": expected_state_dimension_resolution(),
        "capability_contract": {
            "fresh_inputs_allowed": False,
            "environment_step_allowed": False,
            "outcome_inputs_allowed": False,
            "execution_authorized": False,
            "transfer_claim_authorized": False,
            "maximum_authorization": "forward_only",
        },
    }

    return manifest, actions, prefixes, array_sha256(prefixes[0])


def test_complete_preflight_only_authorizes_forward(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest, actions, prefixes, prefix_sha = bundle
    result = run_preflight(manifest)
    assert result["status"] == "passed_forward_only"
    assert result["actor_id"] == ACTOR_ID
    assert result["authorization"] == "forward_only"
    assert result["data_blind"] is True
    assert result["environment_execution_authorized"] is False
    assert result["transfer_claim_authorized"] is False
    semantics = result["static_contract"]["static_semantics"]
    assert semantics["observed_state_dimension_conflict"] is True
    assert semantics["quantile_statistics_available"] is False
    assert result["candidate_validation"]["shape"] == [4, 50, 14]
    assert result["shared_prefix_validation"]["bit_exact_across_candidates"] is True


def test_state_6_vs_stats_env_14_is_blocked_by_default(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest = copy.deepcopy(bundle[0])
    manifest["state_dimension_resolution"] = None
    with pytest.raises(StateDimensionConflict, match="blocked by default"):
        validate_static_preflight(manifest)


def test_14d_identity_shortcut_is_rejected(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest = copy.deepcopy(bundle[0])
    manifest["slot_mapping_contract"]["mode"] = "14d_identity"  # type: ignore[index]
    with pytest.raises(PreflightError, match="identity-by-dimension"):
        validate_static_preflight(manifest)


def test_slot_reordering_is_rejected(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest = copy.deepcopy(bundle[0])
    mapping = manifest["slot_mapping_contract"]["mapping"]  # type: ignore[index]
    mapping[0], mapping[1] = mapping[1], mapping[0]
    with pytest.raises(PreflightError, match="mapping/order"):
        validate_static_preflight(manifest)


@pytest.mark.parametrize("failure", ["shape", "nan", "limit", "degenerate"])
def test_candidate_action_fail_closed_cases(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str], failure: str
) -> None:
    actions = bundle[1].copy()
    if failure == "shape":
        actions = actions[:, :-1]
        match = "shape"
    elif failure == "nan":
        actions[2, 3, 4] = np.nan
        match = "NaN"
    elif failure == "limit":
        actions[1, 0, 4] = 1.2201
        match = "limit violation"
    else:
        actions[:] = actions[0]
        match = "degenerate"
    with pytest.raises(PreflightError, match=match):
        validate_candidate_actions(actions)


def test_shared_prefix_requires_bit_identity_and_claimed_hash(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]
) -> None:
    prefixes = bundle[2].copy()
    prefixes[3, 1] += 1.0
    with pytest.raises(PreflightError, match="bit-exact"):
        validate_shared_prefix(prefixes, claimed_sha256=bundle[3])
    with pytest.raises(PreflightError, match="SHA256"):
        validate_shared_prefix(bundle[2], claimed_sha256="0" * 64)


def test_artifact_sha_tamper_is_rejected(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest = copy.deepcopy(bundle[0])
    manifest["artifacts"]["checkpoint_config"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(PreflightError, match="SHA256 mismatch"):
        validate_static_preflight(manifest)


def test_fresh_artifact_and_array_paths_are_rejected(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str], tmp_path: Path
) -> None:
    manifest = copy.deepcopy(bundle[0])
    fresh_dir = tmp_path / "Fresh50"
    fresh_dir.mkdir()
    source = Path(manifest["artifacts"]["checkpoint_config"]["path"])  # type: ignore[index]
    copied = fresh_dir / "config.json"
    shutil.copy2(source, copied)
    manifest["artifacts"]["checkpoint_config"] = _artifact(copied)  # type: ignore[index]
    with pytest.raises(PreflightError, match="Fresh path"):
        validate_static_preflight(manifest)
    with pytest.raises(PreflightError, match="Fresh path"):
        reject_fresh_path(fresh_dir / "candidate_actions.npy", "candidate_actions")


def test_actor_id_cannot_be_upgraded_to_piper_claim(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    manifest = copy.deepcopy(bundle[0])
    manifest["actor_id"] = "smolvla-piper"
    with pytest.raises(PreflightError, match="actor_id must be exactly"):
        validate_static_preflight(manifest)


def test_capabilities_cannot_authorize_execution_or_outcomes(bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]) -> None:
    for field in (
        "fresh_inputs_allowed",
        "environment_step_allowed",
        "outcome_inputs_allowed",
        "execution_authorized",
        "transfer_claim_authorized",
    ):
        manifest = copy.deepcopy(bundle[0])
        manifest["capability_contract"][field] = True  # type: ignore[index]
        with pytest.raises(PreflightError, match="data-blind and forward-only"):
            validate_static_preflight(manifest)


def test_mapping_is_named_ordinal_angle_preserving_not_equivalence() -> None:
    contract = expected_slot_mapping_contract()
    assert contract["mode"] == "explicit_named_ordinal_angle_preserving_mapping"
    assert contract["angle_values_preserved"] is True
    assert contract["joint_axes_or_kinematics_equivalent"] is False
    assert contract["derived_from_equal_dimension"] is False
    assert contract["physical_equivalence_claimed"] is False
    assert [(row["side"], row["ordinal"]) for row in contract["mapping"]] == [
        (side, ordinal)
        for side in ("left", "right")
        for ordinal in range(1, 8)
    ]


def test_adapter_executes_named_mapping_without_execution_or_scale(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]
) -> None:
    source = bundle[1]
    mapped, receipt = adapt_aloha_source_actions_to_piper_forward_interface(source)
    assert np.array_equal(mapped, source)
    assert not np.shares_memory(mapped, source)
    assert receipt["identity_inferred_from_equal_dimension"] is False
    assert receipt["angle_values_preserved"] is True
    assert receipt["body_specific_gripper_scale_applied"] is False
    assert receipt["kinematic_equivalence_claimed"] is False
    assert receipt["physical_equivalence_claimed"] is False
    assert receipt["execution_authorized"] is False
    assert [row["source_feature_name"] for row in receipt["mapping"]] == list(
        expected_slot_mapping_contract()["mapping"][index]["source_feature_name"]
        for index in range(14)
    )


def test_body_config_must_bind_real_assets_layout_and_declared_urdf(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str], tmp_path: Path
) -> None:
    manifest = copy.deepcopy(bundle[0])
    piper_config = Path(manifest["artifacts"]["piper_body_config"]["path"])  # type: ignore[index]
    value = yaml.safe_load(piper_config.read_text(encoding="utf-8"))
    value["urdf_path"] = "./urdf/piper_description.urdf"
    yaml.safe_dump(value, piper_config.open("w", encoding="utf-8"))
    manifest["artifacts"]["piper_body_config"] = _artifact(piper_config)  # type: ignore[index]
    with pytest.raises(PreflightError, match="urdf_path"):
        validate_static_preflight(manifest)


def test_model_weights_are_authenticated_and_colocated(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str], tmp_path: Path
) -> None:
    manifest = copy.deepcopy(bundle[0])
    foreign = tmp_path / "foreign" / "model.safetensors"
    foreign.parent.mkdir()
    foreign.write_bytes(b"same-looking-foreign-weights")
    manifest["artifacts"]["model_weights"] = _artifact(foreign)  # type: ignore[index]
    with pytest.raises(PreflightError, match="one actor directory"):
        validate_static_preflight(manifest)


def test_probe_receipt_must_record_runtime_state14_and_proper_preprocessing(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]
) -> None:
    manifest = copy.deepcopy(bundle[0])
    probe_path = Path(manifest["probe_artifacts"]["forward_probe_receipt"]["path"])  # type: ignore[index]
    receipt = json.loads(probe_path.read_text(encoding="utf-8"))
    del receipt["runtime_observation_state_dim"]
    _json(probe_path, receipt)
    manifest["probe_artifacts"]["forward_probe_receipt"] = _artifact(probe_path)  # type: ignore[index]
    with pytest.raises(PreflightError, match="runtime_observation_state_dim"):
        run_preflight(manifest)


def test_probe_arrays_are_bound_by_file_and_array_hash(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]
) -> None:
    manifest = copy.deepcopy(bundle[0])
    action_path = Path(manifest["probe_artifacts"]["candidate_actions"]["path"])  # type: ignore[index]
    changed = np.load(action_path, allow_pickle=False)
    changed[1, 0, 0] += 0.001
    np.save(action_path, changed, allow_pickle=False)
    manifest["probe_artifacts"]["candidate_actions"] = _artifact(action_path)  # type: ignore[index]
    with pytest.raises(PreflightError, match="array_outputs"):
        run_preflight(manifest)


def test_freezer_discovers_only_real_robotwin_asset_paths(
    bundle: tuple[dict[str, object], np.ndarray, np.ndarray, str]
) -> None:
    expected = bundle[0]
    artifacts = expected["artifacts"]  # type: ignore[index]
    probes = expected["probe_artifacts"]  # type: ignore[index]
    model_path = Path(artifacts["checkpoint_config"]["path"]).parent
    piper_config = Path(artifacts["piper_body_config"]["path"])
    robotwin_root = piper_config.parents[3]
    frozen = build_manifest(
        model_path=model_path,
        robotwin_root=robotwin_root,
        forward_probe_receipt=Path(probes["forward_probe_receipt"]["path"]),
        candidate_actions=Path(probes["candidate_actions"]["path"]),
        shared_prefixes=Path(probes["shared_prefixes"]["path"]),
        source_image=Path(probes["source_image"]["path"]),
    )
    assert frozen == expected
    assert run_preflight(frozen)["status"] == "passed_forward_only"


def test_freezer_pair_is_created_read_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    _write_pair(
        manifest_path,
        {"kind": "manifest"},
        receipt_path,
        {"kind": "receipt"},
    )
    assert json.loads(manifest_path.read_text()) == {"kind": "manifest"}
    assert json.loads(receipt_path.read_text()) == {"kind": "receipt"}
    assert manifest_path.stat().st_mode & 0o222 == 0
    assert receipt_path.stat().st_mode & 0o222 == 0
