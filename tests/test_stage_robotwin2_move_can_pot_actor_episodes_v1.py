from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import stage_robotwin2_move_can_pot_actor_episodes_v1 as stage  # noqa: E402
import robotwin2_cross_body_canonical_adapter_v1 as adapter  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _instruction(valid: bool = True) -> bytes:
    value = {
        "seen": [f"seen instruction {index}" for index in range(100)],
        "unseen": [f"unseen instruction {index}" for index in range(100)],
    }
    if not valid:
        value.pop("unseen")
    return json.dumps(value).encode("utf-8")


def _write_hdf5(path: Path, action_dim: int = 14, *, external_link: bool = False) -> None:
    frames = 3
    arm_dim = (action_dim - 2) // 2
    with h5py.File(path, "w") as handle:
        endpose = handle.create_group("endpose")
        endpose.create_dataset("left_endpose", data=np.zeros((frames, 7)))
        endpose.create_dataset("left_gripper", data=np.zeros(frames))
        endpose.create_dataset("right_endpose", data=np.zeros((frames, 7)))
        endpose.create_dataset("right_gripper", data=np.zeros(frames))
        joint = handle.create_group("joint_action")
        joint.create_dataset("left_arm", data=np.zeros((frames, arm_dim)))
        joint.create_dataset("left_gripper", data=np.zeros(frames))
        joint.create_dataset("right_arm", data=np.zeros((frames, arm_dim)))
        joint.create_dataset("right_gripper", data=np.zeros(frames))
        joint.create_dataset("vector", data=np.zeros((frames, action_dim)))
        camera = handle.create_group("observation").create_group("head_camera")
        camera.create_dataset("rgb", data=np.asarray([b"jpeg"] * frames, dtype="S4"))
        handle.create_dataset("pointcloud", data=np.zeros((frames, 0)))
        if external_link:
            handle["external"] = h5py.ExternalLink("outside.hdf5", "/data")


def _archive(
    tmp_path: Path,
    *,
    body: str = "piper",
    count: int = 2,
    valid_instruction: bool = True,
    include_pickle: bool = True,
    unsafe_member: bool = False,
    action_dim: int | None = None,
    external_link: bool = False,
) -> tuple[Path, dict]:
    name = f"{body}_clean_50"
    archive_path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/seed.txt", "0\n")
        archive.writestr(f"{name}/scene_info.json", "{}")
        if unsafe_member:
            archive.writestr("../escape.txt", "no")
        for episode_id in range(count):
            hdf_path = tmp_path / f"source_{body}_{episode_id}.hdf5"
            _write_hdf5(
                hdf_path,
                action_dim=(
                    stage.EXPECTED_ACTION_DIMS[body]
                    if action_dim is None
                    else action_dim
                ),
                external_link=external_link,
            )
            archive.write(hdf_path, f"{name}/data/episode{episode_id}.hdf5")
            archive.writestr(
                f"{name}/instructions/episode{episode_id}.json",
                _instruction(valid_instruction),
            )
            archive.writestr(f"{name}/video/episode{episode_id}.mp4", b"opaque-video")
            if include_pickle:
                archive.writestr(
                    f"{name}/_traj_data/episode{episode_id}.pkl",
                    b"opaque-pickle-never-opened",
                )
            hdf_path.unlink()
    return archive_path, {
        "body": body,
        "condition": "clean",
        "source_condition": "clean_50",
        "episode_count": count,
        "path": f"dataset/move_can_pot/{archive_path.name}",
        "size_bytes": archive_path.stat().st_size,
        "payload_sha256": _sha(archive_path),
    }


def _stage(tmp_path: Path, archive: Path, binding: dict) -> tuple[Path, dict]:
    build = tmp_path / "build"
    work = build / ".episode_work"
    work.mkdir(parents=True)
    result = stage.stage_archive(archive, binding, build, work)
    return build, result


def test_streams_hdf5_and_json_only_with_metadata_audit(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path)
    build, result = _stage(tmp_path, archive, binding)

    assert result["episode_count_staged"] == 2
    assert result["frame_count_total"] == 6
    assert result["action_dim"] == 14
    assert result["archive_rehashed_before_member_access"] is True
    for row in result["episodes"]:
        assert row["hdf5_audit"]["metadata_only"] is True
        assert row["hdf5_audit"]["dataset_values_read"] is False
        assert row["instruction_audit"]["seen_count"] == 100
        assert row["instruction_audit"]["instruction_text_selected"] is False
        assert (build / row["hdf5_path"]).is_file()
        assert (build / row["instruction_json_path"]).is_file()
        assert stat.S_IMODE((build / row["hdf5_path"]).stat().st_mode) == 0o444
    staged_names = [path.name for path in build.rglob("*") if path.is_file()]
    assert not any(name.endswith((".pkl", ".mp4")) for name in staged_names)


def test_franka_sixteen_dimensional_action_is_preserved(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, body="franka", count=1)
    _, result = _stage(tmp_path, archive, binding)
    assert result["action_dim"] == 16
    assert result["episodes"][0]["hdf5_audit"]["action_dim"] == 16


def test_wrong_body_action_dimension_fails_closed(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1, action_dim=16)
    with pytest.raises(stage.ActorEpisodeStagingError, match="action dimension"):
        _stage(tmp_path, archive, binding)


def test_missing_pickle_pair_fails_before_member_extraction(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1, include_pickle=False)
    with pytest.raises(stage.ActorEpisodeStagingError, match="exact paired"):
        _stage(tmp_path, archive, binding)
    assert not list((tmp_path / "build").rglob("*.hdf5"))


def test_unsafe_zip_member_fails_without_escape(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1, unsafe_member=True)
    with pytest.raises(stage.ActorEpisodeStagingError, match="safe relative"):
        _stage(tmp_path, archive, binding)
    assert not (tmp_path / "escape.txt").exists()


def test_invalid_instruction_fails_closed(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1, valid_instruction=False)
    with pytest.raises(stage.ActorEpisodeStagingError, match="seen/unseen"):
        _stage(tmp_path, archive, binding)


def test_hdf5_external_link_is_rejected_without_following_it(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1, external_link=True)
    with pytest.raises(stage.ActorEpisodeStagingError, match="external or soft"):
        _stage(tmp_path, archive, binding)


def test_archive_tamper_after_binding_fails_before_member_access(tmp_path: Path) -> None:
    archive, binding = _archive(tmp_path, count=1)
    with archive.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(stage.ActorEpisodeStagingError, match="size changed"):
        _stage(tmp_path, archive, binding)


def test_exact_preregistration_selects_five_clean_archives() -> None:
    preregistration = stage.prereg.build_preregistration()
    source_files = preregistration["official_source_slice"]["files"]
    receipt = {
        "files": [
            {
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "observed_payload_sha256": row["lfs_sha256"],
                "payload_sha256_match": True,
            }
            for row in source_files
        ]
    }
    selected = stage.select_archive_bindings(preregistration, receipt, ["clean"])
    assert [row["body"] for row in selected] == list(stage.BODIES)
    assert all(row["episode_count"] == 50 for row in selected)
    assert all(row["source_condition"] == "clean_50" for row in selected)


def test_manifest_rejects_training_authority_or_tampering() -> None:
    unsigned = {
        "format": stage.FORMAT,
        "status": stage.STATUS,
        "task": stage.TASK,
        "preregistration_sha256": stage.PREREGISTRATION_SHA256,
        "materialization_receipt_sha256": "1" * 64,
        "preregistration_file_sha256": "2" * 64,
        "materialization_receipt_file_sha256": "3" * 64,
        "archive_count": 1,
        "episode_count": 1,
        "archives": [{"episode_count_staged": 1}],
        "read_boundary": {
            "pickle_members_opened": 0,
            "video_members_opened": 0,
            "hdf5_dataset_values_read": False,
            "bulk_archive_extraction_used": False,
        },
        "canonical_adapter_interface": {
            **adapter.contract(),
            "implementation_file": Path(adapter.__file__).name,
            "implementation_file_sha256": stage.file_sha256(Path(adapter.__file__)),
            "action_effect14_materialized_by_this_staging_run": False,
            "state27_materialized_by_this_staging_run": False,
        },
        "authority": {
            "actor_training_data_staging_complete": True,
            "actor_training_authorized": False,
            "critic_or_shared_event_head_training_authorized": False,
            "success_failure_recovery_object_event_supervision_generated": False,
            "task_success_or_cross_embodiment_claim_authorized": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": stage.canonical_sha256(unsigned)}
    stage.validate_manifest(manifest)

    changed = copy.deepcopy(manifest)
    changed["authority"]["actor_training_authorized"] = True
    changed_unsigned = dict(changed)
    changed_unsigned.pop("manifest_sha256")
    changed["manifest_sha256"] = stage.canonical_sha256(changed_unsigned)
    with pytest.raises(stage.ActorEpisodeStagingError, match="data-only boundary"):
        stage.validate_manifest(changed)


def test_existing_output_path_is_create_once(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        stage._output_path(output)


def test_end_to_end_build_publishes_one_read_only_create_once_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_archive, binding = _archive(tmp_path, count=1)
    download = tmp_path / "download"
    archive_path = download / binding["path"]
    archive_path.parent.mkdir(parents=True)
    source_archive.rename(archive_path)
    monkeypatch.setattr(
        stage,
        "select_archive_bindings",
        lambda preregistration, receipt, conditions: [binding],
    )
    receipt = {
        "materialization_receipt_sha256": "a" * 64,
        "download_root": str(download),
    }
    source_files = {
        "preregistration_file_sha256": "b" * 64,
        "materialization_receipt_file_sha256": "c" * 64,
        "preregistration_validation_sha256": "d" * 64,
    }
    output = tmp_path / "published"
    manifest = stage.build_staging(
        {}, receipt, source_files, download, output, ["clean"]
    )

    stage.validate_manifest(manifest)
    assert output.is_dir()
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE((output / "actor_staging_manifest.json").stat().st_mode) == 0o444
    assert manifest["read_boundary"]["pickle_members_opened"] == 0
    assert manifest["canonical_adapter_interface"][
        "action_effect14_materialized_by_this_staging_run"
    ] is False
    with pytest.raises(FileExistsError):
        stage.build_staging({}, receipt, source_files, download, output, ["clean"])


def test_source_contains_no_pickle_deserializer_or_bulk_extract_call() -> None:
    source = Path(stage.__file__).read_text(encoding="utf-8")
    forbidden = (
        "pickle.load",
        "pickle.loads",
        "torch.load",
        "numpy.load",
        "np.load",
        "extractall(",
        ".extract(",
    )
    assert all(token not in source for token in forbidden)


def test_task_space_adapter_produces_body_independent_fourteen_dimensional_effect() -> None:
    half = np.sqrt(0.5)
    left = np.asarray(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [1, 2, 3, half, 0, 0, half],
        ],
        dtype=np.float64,
    )
    right = np.asarray(
        [
            [4, 5, 6, 1, 0, 0, 0],
            [3, 5, 8, -1, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    result = adapter.task_space_action_effect14(
        left, np.asarray([0.2, 0.5]), right, np.asarray([[0.8], [0.6]])
    )

    assert result.shape == (1, 14)
    np.testing.assert_allclose(result[0, :3], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(result[0, 3:6], [0, 0, np.pi / 2], atol=1e-6)
    np.testing.assert_allclose(result[0, 6], 0.3, atol=1e-6)
    np.testing.assert_allclose(result[0, 7:10], [-1, 0, 2], atol=1e-6)
    np.testing.assert_allclose(result[0, 10:13], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result[0, 13], -0.2, atol=1e-6)


def test_state27_packer_freezes_requested_channel_order() -> None:
    parts = [
        np.full((2, width), index, dtype=np.float32)
        for index, width in enumerate((3, 3, 3, 3, 2, 4, 5, 4), start=1)
    ]
    parts[5] = np.asarray([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=np.float32)
    result = adapter.pack_shared_critic_state27(*parts)
    assert result.shape == (2, 27)
    np.testing.assert_array_equal(
        result[0],
        np.asarray(
            [1] * 3
            + [2] * 3
            + [3] * 3
            + [4] * 3
            + [5] * 2
            + [1, 0, 0, 0]
            + [7] * 5
            + [8] * 4,
            dtype=np.float32,
        ),
    )
    contract = adapter.contract()
    assert contract["action_effect14_channels"] == list(adapter.ACTION_EFFECT14_CHANNELS)
    assert contract["object_effect6_channels"] == list(adapter.OBJECT_EFFECT6_CHANNELS)
    assert contract["state27_channels"] == list(adapter.STATE27_CHANNELS)
    assert contract["success_failure_recovery_object_event_labels_generated"] is False


def test_public_object_rotation_effect_uses_shortest_wxyz_axis_angle() -> None:
    identity = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    half = np.pi / 4.0
    quarter_turn_z = np.asarray(
        [[np.cos(half), 0.0, 0.0, np.sin(half)]], dtype=np.float64
    )
    result = adapter.relative_axis_angle_wxyz(identity, quarter_turn_z)
    np.testing.assert_allclose(result, [[0.0, 0.0, np.pi / 2.0]], atol=1e-7)
    # Quaternion sign represents the same rotation and must not change the
    # canonical shortest-axis-angle effect.
    np.testing.assert_allclose(
        adapter.relative_axis_angle_wxyz(identity, -quarter_turn_z),
        result,
        atol=1e-7,
    )


def test_canonical_adapter_rejects_mismatched_or_nonfinite_inputs() -> None:
    pose = np.asarray([[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=np.float64)
    with pytest.raises(adapter.CanonicalAdapterError, match="right_gripper"):
        adapter.task_space_action_effect14(pose, [0, 0], pose, [0])
    with pytest.raises(adapter.CanonicalAdapterError, match="finite"):
        adapter.pack_shared_critic_state27(
            np.asarray([[np.nan, 0, 0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 2)),
            np.zeros((1, 4)),
            np.zeros((1, 5)),
            np.zeros((1, 4)),
        )
