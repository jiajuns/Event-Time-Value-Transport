from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_smolvla_piper_schema6_training_manifest_v2 as aggregate  # noqa: E402
import preregister_smolvla_piper_schema6_multiseed_collection_v2 as prereg  # noqa: E402
import train_smolvla_piper_schema6_embodiment_adapter as trainer  # noqa: E402
from etsf_schema6_pose_quality import registry_sha256, spec_sha256  # noqa: E402


EVENT_SHA = hashlib.sha256(b"event-spec").hexdigest()
COLLECTOR_SHA = hashlib.sha256(b"collector-lineage").hexdigest()


def _canonical(value: object) -> str:
    return aggregate.canonical_sha256(value)


def _signed(value: dict[str, object], key: str) -> dict[str, object]:
    result = dict(value)
    result[key] = _canonical(result)
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _semantic_receipt() -> dict[str, object]:
    return _signed(
        {"format": "instruction_semantics_v1", "instruction": prereg.INSTRUCTION},
        "receipt_sha256",
    )


def _target_manifest(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    semantic = _semantic_receipt()
    instruction_sha = hashlib.sha256(prereg.INSTRUCTION.encode()).hexdigest()
    splits: dict[str, list[dict[str, object]]] = {}
    all_rows: list[dict[str, object]] = []
    global_index = 0
    for split, count in prereg.SPLIT_COUNTS.items():
        rows: list[dict[str, object]] = []
        for ordinal in range(count):
            row: dict[str, object] = {
                "task": prereg.TASK,
                "actor_id": prereg.ACTOR_ID,
                "target_body": prereg.TARGET_BODY,
                "global_ordinal": global_index,
                "split": split,
                "ordinal": ordinal,
                "stage_role": "identity_only_reset",
                "requested_seed": 100_201_000 + global_index,
                "resolved_seed": 200_201_000 + global_index,
                "instruction": prereg.INSTRUCTION,
                "instruction_sha256": instruction_sha,
                "instruction_semantics_receipt": semantic,
                "instruction_semantics_receipt_sha256": semantic["receipt_sha256"],
                "initial_scene_state_sha256": hashlib.sha256(f"scene-{global_index}".encode()).hexdigest(),
                "initial_measured_joint_state_sha256": hashlib.sha256(f"joint-{global_index}".encode()).hexdigest(),
                "initial_commanded_drive_target_sha256": hashlib.sha256(f"drive-{global_index}".encode()).hexdigest(),
            }
            row["pair_id"] = prereg.canonical_sha256(prereg._row_pair_identity(row))
            rows.append(row)
            all_rows.append(row)
            global_index += 1
        splits[split] = rows
    value = _signed(
        {
            "format": prereg.TARGET_FORMAT,
            "status": prereg.TARGET_STATUS,
            "task": prereg.TASK,
            "actor_id": prereg.ACTOR_ID,
            "target_body": prereg.TARGET_BODY,
            "splits": splits,
            "capability_receipt": {
                "policy_execution_authorized_by_manifest": False,
                "labels_or_outcomes_read": False,
            },
        },
        "seed_manifest_sha256",
    )
    _write_json(path, value)
    path.chmod(0o444)
    return value, all_rows


def _registry(index: int, *, bad_asset: bool = False) -> dict[str, object]:
    can_asset = "wrong/base0" if bad_asset else f"105_sauce-can/base{index % 4}"
    return {
        "format": "etsf_schema6_object_registry_v1",
        "objects": [
            {
                "name": "can",
                "stable_sim_actor_id": f"task_attr=can;sapien_actor_name=can-{index}",
                "asset_model_id": can_asset,
                "role": "manipulated",
                "is_static": False,
            },
            {
                "name": "pot",
                "stable_sim_actor_id": f"task_attr=pot;sapien_actor_name=pot-{index}",
                "asset_model_id": f"060_kitchenpot/base{index % 3}",
                "role": "receptacle",
                "is_static": False,
            },
        ],
    }


def _pose_spec(registry: dict[str, object]) -> dict[str, object]:
    return {
        "format": "etsf_schema6_pose_quality_spec_v1",
        "schema_version": 6,
        "object_registry_sha256": registry_sha256(registry),
        "pose_layout": {
            "shape_suffix": [7],
            "translation_indices": [0, 1, 2],
            "quaternion_indices": [3, 4, 5, 6],
            "quaternion_order": "wxyz",
            "frame": "simulator_world",
            "translation_unit": "metre",
            "rotation_unit": "radian",
        },
        "time_layout": {
            "timestamp_unit": "second",
            "timestamp_clock": "simulator_monotonic",
            "control_step_semantics": "sample_after_completed_control_step",
            "physics_substep_semantics": "substeps_since_previous_sample_zero_at_reset",
        },
        "thresholds": {
            "world_aabb_m": [[-3.0, 3.0], [-1.0, 2.0], [-0.5, 3.0]],
            "quaternion_norm_abs_tolerance": 1e-3,
            "max_step_translation_m": 7.5,
            "max_step_rotation_rad": math.pi,
            "static_object_max_step_translation_m": 1e-6,
            "static_object_max_step_rotation_rad": 1e-6,
            "timestamp_step_min_s": 0.004,
            "timestamp_step_max_s": 4.0,
            "max_physics_substeps_per_control_step": 1000,
        },
        "threshold_basis": {
            "thresholds_fit_from_pose_data": False,
            "source": "synthetic task geometry and simulator timing only",
            "frozen_before_collection": True,
        },
    }


def _freeze_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _native_v2_collection_root(
    parent: Path, row: dict[str, object], *, index: int
) -> tuple[Path, dict[str, object]]:
    root = parent / f"native_seed_{index:03d}"
    root.mkdir(parents=True)
    registry = _registry(index)
    pose = _pose_spec(registry)
    _write_json(root / "object_registry.json", registry)
    _write_json(root / "pose_quality_spec.json", pose)
    group = root / "schema6_group.hdf5"
    group.write_bytes(f"opaque-native-v2-{index}".encode())
    preregistration_sha = "8" * 64
    command: dict[str, object] = {
        "split": row["split"],
        "ordinal": row["ordinal"],
        "requested_seed": row["requested_seed"],
        "expected_resolved_seed": row["resolved_seed"],
        "pair_id": row["pair_id"],
        "expected_initial_scene_state_sha256": row["initial_scene_state_sha256"],
        "candidate_original_indices": [0, 1, 2, 3],
        "argv": ["/bound/python", "/bound/runner.py", "collect-one"],
        "outputs": {
            "seed_root": str(root.resolve()),
            "per_seed_reset_receipt": str((root / "per_seed_reset_receipt.json").resolve()),
            "group_hdf5": str(group.resolve()),
            "completed_group_receipt": str((root / "completed_group_receipt.json").resolve()),
        },
        "bindings": {
            "target_seed_manifest_file_sha256": "6" * 64,
            "target_seed_manifest_sha256": "7" * 64,
            "r6j_code_closure_sha256": COLLECTOR_SHA,
            "event_spec_sha256": EVENT_SHA,
            "runtime_python_sha256": "9" * 64,
            "v2_runner_sha256": "a" * 64,
        },
    }
    command["command_sha256"] = _canonical(command)
    reset = _signed(
        {
            "format": aggregate.RESET_RECEIPT_FORMAT,
            "status": aggregate.RESET_RECEIPT_STATUS,
            "preregistration_sha256": preregistration_sha,
            "command_sha256": command["command_sha256"],
            "split": row["split"],
            "ordinal": row["ordinal"],
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "pair_id": row["pair_id"],
            "initial_scene_state_sha256": row["initial_scene_state_sha256"],
            "initial_measured_joint_state_sha256": row["initial_measured_joint_state_sha256"],
            "initial_commanded_drive_target_sha256": row["initial_commanded_drive_target_sha256"],
            "object_registry_sha256": registry_sha256(registry),
            "pose_spec_sha256": spec_sha256(
                pose, expected_registry_sha256=registry_sha256(registry)
            ),
            "identity_validation_count_before_policy_query": 1,
            "policy_queries_before_reset_receipt": 0,
            "evaluation_execution_authorized": False,
            "protected_inputs_read": False,
        },
        "reset_receipt_sha256",
    )
    receipt = _signed(
        {
            "format": aggregate.GROUP_RECEIPT_FORMAT,
            "status": aggregate.GROUP_RECEIPT_STATUS,
            "preregistration_sha256": preregistration_sha,
            "command_sha256": command["command_sha256"],
            "split": row["split"],
            "ordinal": row["ordinal"],
            "requested_seed": row["requested_seed"],
            "resolved_seed": row["resolved_seed"],
            "pair_id": row["pair_id"],
            "candidate_original_indices": [0, 1, 2, 3],
            "branch_records": 4,
            "per_seed_reset_receipt_sha256": reset["reset_receipt_sha256"],
            "object_registry_sha256": registry_sha256(registry),
            "pose_spec_sha256": spec_sha256(
                pose, expected_registry_sha256=registry_sha256(registry)
            ),
            "group_file_sha256": _file_sha(group),
        },
        "group_receipt_sha256",
    )
    _write_json(root / "per_seed_reset_receipt.json", reset)
    _write_json(root / "completed_group_receipt.json", receipt)
    _freeze_tree(root)
    return root, command


def _collection_root(
    parent: Path,
    row: dict[str, object],
    *,
    index: int,
    target_file_sha: str,
    target_logical_sha: str,
    bad_asset: bool = False,
) -> Path:
    root = parent / f"collection_{row['split']}_{int(row['ordinal']):03d}"
    root.mkdir()
    registry = _registry(index, bad_asset=bad_asset)
    pose = _pose_spec(registry)
    registry_path = root / "object_registry.json"
    pose_path = root / "pose_quality_spec.json"
    group_path = root / "schema6_group.hdf5"
    _write_json(registry_path, registry)
    _write_json(pose_path, pose)
    opaque = f"opaque-not-hdf-{row['split']}-{row['ordinal']}".encode()
    group_path.write_bytes(opaque)
    group_sha = hashlib.sha256(opaque).hexdigest()
    identity = {
        "split": row["split"],
        "ordinal": row["ordinal"],
        "requested_seed": row["requested_seed"],
        "resolved_seed": row["resolved_seed"],
        "pair_id": row["pair_id"],
        "task": aggregate.TASK,
        "body": aggregate.BODY,
        "policy": aggregate.POLICY,
    }
    authority = _signed(
        {
            "format": aggregate.COLLECTION_AUTHORITY_FORMAT,
            "status": "frozen_before_collection",
            "identity": identity,
            "candidate_original_indices": [0, 1, 2, 3],
            "per_seed_live_registry_materialized": True,
            "fixed_seed_registry_reused": False,
            "bindings": {
                "target_seed_manifest_file_sha256": target_file_sha,
                "target_seed_manifest_sha256": target_logical_sha,
                "event_spec_sha256": EVENT_SHA,
                "collector_lineage_sha256": COLLECTOR_SHA,
            },
            "artifacts": {
                "object_registry": {
                    "path": registry_path.name,
                    "file_sha256": _file_sha(registry_path),
                    "logical_sha256": registry_sha256(registry),
                },
                "pose_quality_spec": {
                    "path": pose_path.name,
                    "file_sha256": _file_sha(pose_path),
                    "logical_sha256": spec_sha256(pose, expected_registry_sha256=registry_sha256(registry)),
                },
            },
            "output_contract": {
                "manifest": "manifest.json",
                "group": "schema6_group.hdf5",
                "create_once": True,
            },
            "capability_contract": aggregate.CAPABILITY_CONTRACT,
        },
        "authority_sha256",
    )
    authority_path = root / "collection_authority.json"
    _write_json(authority_path, authority)
    common = {
        "status": aggregate.COLLECTION_STATUS,
        "identity": identity,
        "candidate_original_indices": [0, 1, 2, 3],
        "branch_records": 4,
        "object_registry_sha256": registry_sha256(registry),
        "pose_spec_sha256": spec_sha256(pose, expected_registry_sha256=registry_sha256(registry)),
        "event_spec_sha256": EVENT_SHA,
        "collector_lineage_sha256": COLLECTOR_SHA,
        "capability_contract": aggregate.CAPABILITY_CONTRACT,
    }
    group_record = {"path": group_path.name, "file_sha256": group_sha}
    manifest = _signed(
        {
            "format": aggregate.COLLECTION_MANIFEST_FORMAT,
            **common,
            "group": group_record,
        },
        "manifest_sha256",
    )
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    final = _signed(
        {
            "format": aggregate.COLLECTION_FINAL_FORMAT,
            **common,
            "authority": {
                "path": authority_path.name,
                "file_sha256": _file_sha(authority_path),
                "logical_sha256": authority["authority_sha256"],
            },
            "manifest": {
                "path": manifest_path.name,
                "file_sha256": _file_sha(manifest_path),
                "logical_sha256": manifest["manifest_sha256"],
            },
            "group": group_record,
            "hdf5_content_opened_by_finalizer": False,
            "labels_read_by_finalizer": False,
        },
        "receipt_sha256",
    )
    _write_json(root / "final_receipt.json", final)
    _freeze_tree(root)
    return root


def _base(tmp_path: Path) -> tuple[Path, dict[str, object], list[dict[str, object]], dict[str, object]]:
    target_path = tmp_path / "target_seed_manifest.json"
    target, rows = _target_manifest(target_path)
    kwargs: dict[str, object] = {
        "target_seed_manifest_path": target_path,
        "expected_target_manifest_file_sha256": _file_sha(target_path),
        "expected_target_manifest_logical_sha256": target["seed_manifest_sha256"],
        "event_spec_sha256": EVENT_SHA,
        "collector_lineage_sha256": COLLECTOR_SHA,
        "bound_trainer_path": Path(trainer.__file__).resolve(),
        "expected_bound_trainer_sha256": _file_sha(Path(trainer.__file__).resolve()),
        "collection_roots": [],
        "output_directory": tmp_path,
    }
    return target_path, target, rows, kwargs


def test_insufficient_receipt_is_signed_and_never_opens_opaque_hdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, rows, kwargs = _base(tmp_path)
    root = _collection_root(
        tmp_path, rows[0], index=0,
        target_file_sha=kwargs["expected_target_manifest_file_sha256"],
        target_logical_sha=target["seed_manifest_sha256"],
    )
    kwargs["collection_roots"] = [root]
    original_open = Path.open
    hdf_touches: list[Path] = []

    def guarded_open(self: Path, *args: object, **options: object):
        if self.suffix.casefold() in aggregate.HDF_SUFFIXES:
            hdf_touches.append(self)
            raise AssertionError("opaque HDF must not be opened")
        return original_open(self, *args, **options)

    monkeypatch.setattr(Path, "open", guarded_open)
    receipt = aggregate.aggregate(**kwargs)
    assert receipt["status"] == aggregate.INSUFFICIENT_STATUS
    assert receipt["training_authorized"] is False
    assert receipt["present_group_count"] == 1
    assert receipt["missing_group_count"] == 129
    assert receipt["receipt_sha256"] == _canonical(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert hdf_touches == []
    assert not (tmp_path / "schema6_training_manifest_v2_compat.json").exists()


def test_complete_130_outputs_trainer_manifest_and_leak_free_external_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, rows, kwargs = _base(tmp_path)
    selected = rows[:130]
    roots = [
        _collection_root(
            tmp_path, row, index=index,
            target_file_sha=kwargs["expected_target_manifest_file_sha256"],
            target_logical_sha=target["seed_manifest_sha256"],
        )
        for index, row in enumerate(selected)
    ]
    kwargs["collection_roots"] = list(reversed(roots))
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **options: object):
        if self.suffix.casefold() in aggregate.HDF_SUFFIXES:
            raise AssertionError("opaque HDF must not be opened")
        return original_open(self, *args, **options)

    monkeypatch.setattr(Path, "open", guarded_open)
    receipt = aggregate.aggregate(**kwargs)
    assert receipt["status"] == aggregate.COMPLETE_STATUS
    assert receipt["hdf5_content_files_opened"] == 0
    assert receipt["direct_bound_trainer_execution_authorized"] is True
    manifest = json.loads((tmp_path / "schema6_training_manifest_v2_compat.json").read_text())
    assert manifest["format"] == aggregate.TRAINER_MANIFEST_FORMAT
    assert len(manifest["groups"]) == 130
    assert [row["requested_seed"] for row in manifest["groups"]] == [
        row["requested_seed"] for row in selected
    ]
    scanned, descriptors = trainer.scan_manifest(
        (tmp_path / "schema6_training_manifest_v2_compat.json").resolve()
    )
    assert scanned["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(descriptors) == 130
    split = json.loads((tmp_path / "schema6_external_group_split_v2.json").read_text())
    partition = json.loads((tmp_path / "schema6_target_partition_v2.json").read_text())
    assert (len(split["train"]), len(split["validation"]), len(split["test"])) == (60, 20, 50)
    assert set(split["test"]) == set(partition["validation"])
    assert not (set(split["train"]) | set(split["validation"])) & set(partition["validation"])
    assert partition["evaluation"] == []
    expected_record = receipt["expected_manifest_split_receipt"]
    bound_split, split_audit = trainer.validate_external_split_authority(
        expected_receipt_path=Path(expected_record["path"]),
        expected_receipt_file_sha256=expected_record["file_sha256"],
        manifest_path=(tmp_path / "schema6_training_manifest_v2_compat.json").resolve(),
        manifest=manifest,
        descriptors=descriptors,
    )
    assert bound_split["split_sha256"] == split["split_sha256"]
    assert split_audit["sealed_test_hdf5_files_opened"] == 0
    with pytest.raises(trainer.AdapterContractError, match="receipt file SHA256 mismatch"):
        trainer.validate_external_split_authority(
            expected_receipt_path=Path(expected_record["path"]),
            expected_receipt_file_sha256="0" * 64,
            manifest_path=(tmp_path / "schema6_training_manifest_v2_compat.json").resolve(),
            manifest=manifest,
            descriptors=descriptors,
        )


def test_native_v2_seed_receipt_is_validated_without_opening_hdf_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _manifest, rows = _target_manifest(tmp_path / "target.json")
    root, command = _native_v2_collection_root(tmp_path, rows[0], index=0)
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.suffix.casefold() in {".h5", ".hdf", ".hdf5"}:
            raise AssertionError("native-v2 HDF bytes were opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    descriptor = aggregate.validate_native_v2_collection_root(
        root,
        expected_command=command,
        expected_preregistration_file_sha256="6" * 64,
        expected_preregistration_logical_sha256="8" * 64,
        expected_event_spec_sha256=EVENT_SHA,
        expected_collector_lineage_sha256=COLLECTOR_SHA,
    )
    assert descriptor.identity["split"] == "adaptation"
    assert descriptor.group_path.name == "schema6_group.hdf5"
    assert len(descriptor.group_file_sha256) == 64


def test_evaluation_collection_root_is_rejected(tmp_path: Path) -> None:
    _, target, rows, kwargs = _base(tmp_path)
    evaluation = rows[130]
    root = _collection_root(
        tmp_path, evaluation, index=130,
        target_file_sha=kwargs["expected_target_manifest_file_sha256"],
        target_logical_sha=target["seed_manifest_sha256"],
    )
    kwargs["collection_roots"] = [root]
    with pytest.raises(aggregate.TrainingManifestContractError, match="evaluation collection root"):
        aggregate.aggregate(**kwargs)


def test_cross_contract_hdf_sha_mismatch_fails_without_hdf_open(tmp_path: Path) -> None:
    _, target, rows, kwargs = _base(tmp_path)
    root = _collection_root(
        tmp_path, rows[0], index=0,
        target_file_sha=kwargs["expected_target_manifest_file_sha256"],
        target_logical_sha=target["seed_manifest_sha256"],
    )
    final_path = root / "final_receipt.json"
    root.chmod(0o755); final_path.chmod(0o644)
    final = json.loads(final_path.read_text())
    final["group"]["file_sha256"] = "f" * 64
    final.pop("receipt_sha256")
    final["receipt_sha256"] = _canonical(final)
    _write_json(final_path, final)
    _freeze_tree(root)
    kwargs["collection_roots"] = [root]
    with pytest.raises(aggregate.TrainingManifestContractError, match="final/manifest lineage"):
        aggregate.aggregate(**kwargs)


def test_wrong_can_asset_registry_is_rejected(tmp_path: Path) -> None:
    _, target, rows, kwargs = _base(tmp_path)
    root = _collection_root(
        tmp_path, rows[0], index=0,
        target_file_sha=kwargs["expected_target_manifest_file_sha256"],
        target_logical_sha=target["seed_manifest_sha256"],
        bad_asset=True,
    )
    kwargs["collection_roots"] = [root]
    with pytest.raises(aggregate.TrainingManifestContractError, match="can actor/asset"):
        aggregate.aggregate(**kwargs)


def test_sensitive_collection_root_rejected_before_metadata_open(tmp_path: Path) -> None:
    _, _, _, kwargs = _base(tmp_path)
    protected = tmp_path / "confirmation_hidden" / "collection"
    kwargs["collection_roots"] = [protected]
    with pytest.raises(aggregate.TrainingManifestContractError, match="forbidden component"):
        aggregate.aggregate(**kwargs)
