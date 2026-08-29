from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_robotwin2_move_can_pot_public_materialization_v1 as verify  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    root = tmp_path / "download"
    scope = root / "dataset" / "move_can_pot"
    scope.mkdir(parents=True)
    files = []
    for name, method in (("piper_clean_50.zip", zipfile.ZIP_STORED), ("piper_randomized_500.zip", zipfile.ZIP_DEFLATED)):
        path = scope / name
        with zipfile.ZipFile(path, "w", compression=method) as archive:
            archive.writestr("episode_000/data.pkl", b"opaque-pickle-bytes-not-deserialized")
            archive.writestr("episode_000/camera.npy", b"opaque-numpy-bytes-not-deserialized")
        files.append(
            {
                "path": f"dataset/move_can_pot/{name}",
                "size_bytes": path.stat().st_size,
                "lfs_sha256": _sha256(path),
            }
        )
    preregistration = {
        "preregistration_sha256": "a" * 64,
        "official_source_slice": {
            "task_path": "dataset/move_can_pot",
            "hf_repo_id": "TianxingChen/RoboTwin2.0",
            "hf_repo_revision": "revision",
            "total_size_bytes": sum(row["size_bytes"] for row in files),
            "files": files,
        },
    }
    audit = {"status": "test_exact_preregistration_validated"}
    return root, preregistration, audit


def test_complete_exact_slice_materializes_with_metadata_only_zip_audit(tmp_path: Path) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    receipt = verify.build_receipt(root, preregistration, audit, "provided_json")

    verify.validate_receipt(receipt)
    assert receipt["materialized"] is True
    assert receipt["official_file_count"] == 2
    assert receipt["no_missing_or_extra_official_task_files"] is True
    assert receipt["all_exact_archive_payload_sha256_verified"] is True
    assert receipt["read_boundary"]["zip_member_payload_bytes_read"] == 0
    assert receipt["read_boundary"]["archive_extracted"] is False
    assert receipt["read_boundary"]["pickle_payload_opened_or_deserialized"] is False
    assert receipt["authority"]["training_authorized"] is False
    assert receipt["authority"]["evaluation_authorized"] is False
    assert receipt["authority"]["cross_embodiment_performance_claim_authorized"] is False
    assert receipt["implementation_binding"]["verifier_file_sha256"] == _sha256(
        Path(verify.__file__)
    )
    assert receipt["implementation_binding"][
        "preregistration_module_file_sha256"
    ] == _sha256(Path(verify.prereg.__file__))
    for row in receipt["files"]:
        central = row["zip_central_directory_audit"]
        assert central["member_count"] == 2
        assert central["member_payload_bytes_read"] == 0
        assert central["member_paths_safe_relative_posix"] is True
        assert central["file_suffix_member_counts"] == {".npy": 1, ".pkl": 1}
        assert central["member_payload_crc_verified"] is False


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_missing_or_extra_official_task_file_fails_before_receipt(tmp_path: Path, change: str) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    if change == "missing":
        (root / preregistration["official_source_slice"]["files"][0]["path"]).unlink()
    else:
        (root / "dataset" / "move_can_pot" / "partial.tmp").write_bytes(b"partial")
    with pytest.raises(verify.PublicMaterializationError, match="inventory mismatch"):
        verify.build_receipt(root, preregistration, audit, "provided_json")


@pytest.mark.parametrize("change", ["size", "sha256"])
def test_wrong_size_or_payload_hash_fails_closed(tmp_path: Path, change: str) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    if change == "size":
        preregistration["official_source_slice"]["files"][0]["size_bytes"] += 1
        preregistration["official_source_slice"]["total_size_bytes"] += 1
        match = "size mismatch"
    else:
        preregistration["official_source_slice"]["files"][0]["lfs_sha256"] = "0" * 64
        match = "SHA-256 mismatch"
    with pytest.raises(verify.PublicMaterializationError, match=match):
        verify.build_receipt(root, preregistration, audit, "provided_json")


def test_symlink_archive_fails_inventory_without_following_target(tmp_path: Path) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    path = root / preregistration["official_source_slice"]["files"][0]["path"]
    target = tmp_path / "outside.zip"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(verify.PublicMaterializationError, match="symbolic link"):
        verify.build_receipt(root, preregistration, audit, "provided_json")


def test_preregistered_path_cannot_escape_official_task_scope(tmp_path: Path) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    preregistration["official_source_slice"]["files"][0]["path"] = (
        "dataset/move_can_pot/../../outside.zip"
    )
    with pytest.raises(verify.PublicMaterializationError, match="path, size, or payload hash"):
        verify.build_receipt(root, preregistration, audit, "provided_json")


@pytest.mark.parametrize("member", ["../escape.pkl", "/absolute.npy", "windows\\escape.pkl"])
def test_unsafe_zip_member_path_fails_without_extraction(tmp_path: Path, member: str) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    row = preregistration["official_source_slice"]["files"][0]
    path = root / row["path"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"opaque")
    row["size_bytes"] = path.stat().st_size
    row["lfs_sha256"] = _sha256(path)
    preregistration["official_source_slice"]["total_size_bytes"] = sum(
        child["size_bytes"] for child in preregistration["official_source_slice"]["files"]
    )
    with pytest.raises(verify.PublicMaterializationError, match="unsafe"):
        verify.build_receipt(root, preregistration, audit, "provided_json")
    assert not (tmp_path / "escape.pkl").exists()


def test_duplicate_zip_members_fail_closed(tmp_path: Path) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    row = preregistration["official_source_slice"]["files"][0]
    path = root / row["path"]
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("episode/data.pkl", b"one")
            archive.writestr("episode/data.pkl", b"two")
    row["size_bytes"] = path.stat().st_size
    row["lfs_sha256"] = _sha256(path)
    preregistration["official_source_slice"]["total_size_bytes"] = sum(
        child["size_bytes"] for child in preregistration["official_source_slice"]["files"]
    )
    with pytest.raises(verify.PublicMaterializationError, match="duplicate"):
        verify.build_receipt(root, preregistration, audit, "provided_json")


def test_create_once_receipt_is_read_only_and_tampering_is_rejected(tmp_path: Path) -> None:
    root, preregistration, audit = _fixture(tmp_path)
    receipt = verify.build_receipt(root, preregistration, audit, "provided_json")
    output = tmp_path / "receipt.json"
    verify.write_json_new(output, receipt)

    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        verify.write_json_new(output, receipt)
    changed = copy.deepcopy(receipt)
    changed["materialized"] = False
    with pytest.raises(verify.PublicMaterializationError, match="canonical SHA"):
        verify.validate_receipt(changed)


def test_reviewed_preregistration_can_be_rebuilt_without_dataset_access() -> None:
    value, audit, source = verify.load_preregistration(None)
    assert source == "deterministic_module_rebuild"
    assert value["official_source_slice"]["tree_entry_count"] == 11
    assert audit["official_file_count"] == 11
    assert audit["archive_payloads_opened"] == 0


def test_source_never_calls_extract_or_payload_deserializers() -> None:
    source = Path(verify.__file__).read_text(encoding="utf-8")
    forbidden = ["extractall(", ".extract(", "pickle.load", "numpy.load", "np.load", "torch.load"]
    assert all(token not in source for token in forbidden)
