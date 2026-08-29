#!/usr/bin/env python3
"""Stage public RoboTwin2 actor episodes without bulk extraction or pickle IO.

The command accepts the immutable five-body preregistration and a verified
public materialization receipt.  It re-hashes each selected archive, streams
one HDF5 episode at a time through a controlled temporary file, audits HDF5
metadata without reading dataset values, validates the paired instruction
JSON, and publishes a create-once staging tree plus a signed manifest.

Only raw HDF5 and instruction JSON members are opened.  Pickle and video
members are checked by central-directory identity but are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence

import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg
import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import verify_robotwin2_move_can_pot_public_materialization_v1 as materialization


FORMAT = "etsf_robotwin2_move_can_pot_actor_episode_staging_manifest_v1"
STATUS = "complete_public_actor_episode_staging_no_training_or_label_authority"
DATASET_REPO = prereg.HF_REPO_ID
DATASET_REVISION = prereg.HF_REPO_REVISION
TASK = prereg.TASK
BODIES = prereg.BODIES
PREREGISTRATION_SHA256 = (
    "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee"
)
CONDITION_CONTRACT = {
    "clean": ("clean_50", 50),
    "randomized": ("randomized_500", 500),
}
EXPECTED_ACTION_DIMS = {
    "aloha-agilex": 14,
    "arx-x5": 14,
    "franka": 16,
    "piper": 14,
    "ur5": 14,
}
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_HDF5_BYTES = 8 * 1024 * 1024 * 1024
MAX_CONTRACT_BYTES = 16 * 1024 * 1024
EPISODE_MEMBER = re.compile(
    r"^(?P<prefix>[^/]+)/(?P<kind>data|instructions|video|_traj_data)/"
    r"episode(?P<episode>[0-9]+)\.(?P<suffix>hdf5|json|mp4|pkl)$"
)
REQUIRED_HDF5_DATASETS = (
    "endpose/left_endpose",
    "endpose/left_gripper",
    "endpose/right_endpose",
    "endpose/right_gripper",
    "joint_action/left_arm",
    "joint_action/left_gripper",
    "joint_action/right_arm",
    "joint_action/right_gripper",
    "joint_action/vector",
    "observation/head_camera/rgb",
)


class ActorEpisodeStagingError(RuntimeError):
    """A source binding, public archive, episode, or output failed closed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_regular(path: Path, label: str) -> tuple[dict[str, Any], str]:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ActorEpisodeStagingError(f"cannot open {label} read-only") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_CONTRACT_BYTES:
            raise ActorEpisodeStagingError(f"{label} must be a bounded regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise ActorEpisodeStagingError(f"{label} changed while being read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActorEpisodeStagingError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ActorEpisodeStagingError(f"{label} must be a JSON object")
    return value, digest.hexdigest()


def _real_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as error:
        raise ActorEpisodeStagingError(f"{label} does not exist") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ActorEpisodeStagingError(f"{label} must be a real directory")
    return absolute


def _safe_relative(path: str, label: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ActorEpisodeStagingError(f"{label} is not a safe relative POSIX path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ActorEpisodeStagingError(f"{label} is not a safe relative POSIX path")
    return pure


def _contained_regular(root: Path, relative: str, label: str) -> Path:
    pure = _safe_relative(relative, label)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as error:
            raise ActorEpisodeStagingError(f"missing parent for {label}") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ActorEpisodeStagingError(f"{label} parent is a symlink or non-directory")
    path = current / pure.name
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ActorEpisodeStagingError(f"missing {label}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ActorEpisodeStagingError(f"{label} must be a real regular file")
    return path


def load_source_contracts(
    preregistration_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preregistration, prereg_file_sha = _read_json_regular(
        preregistration_path, "preregistration"
    )
    receipt, receipt_file_sha = _read_json_regular(receipt_path, "materialization receipt")
    try:
        prereg_audit = prereg.validate_preregistration(preregistration)
    except prereg.CrossEmbodimentSlicePreregistrationError as error:
        raise ActorEpisodeStagingError("preregistration is not the reviewed contract") from error
    if preregistration.get("preregistration_sha256") != PREREGISTRATION_SHA256:
        raise ActorEpisodeStagingError("unexpected five-body preregistration SHA-256")
    try:
        materialization.validate_receipt(receipt)
    except materialization.PublicMaterializationError as error:
        raise ActorEpisodeStagingError("materialization receipt is not valid now") from error
    if (
        receipt.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or receipt.get("hf_repo_id") != DATASET_REPO
        or receipt.get("hf_repo_revision") != DATASET_REVISION
        or receipt.get("official_file_count") != 11
        or receipt.get("all_exact_archive_payload_sha256_verified") is not True
    ):
        raise ActorEpisodeStagingError("receipt does not bind the exact reviewed public slice")
    return preregistration, receipt, {
        "preregistration_file_sha256": prereg_file_sha,
        "materialization_receipt_file_sha256": receipt_file_sha,
        "preregistration_validation_sha256": canonical_sha256(prereg_audit),
    }


def select_archive_bindings(
    preregistration: Mapping[str, Any],
    receipt: Mapping[str, Any],
    conditions: Sequence[str],
) -> list[dict[str, Any]]:
    if not conditions or len(set(conditions)) != len(conditions):
        raise ActorEpisodeStagingError("conditions must be nonempty and unique")
    if any(condition not in CONDITION_CONTRACT for condition in conditions):
        raise ActorEpisodeStagingError("condition is outside clean/randomized contract")
    source = preregistration.get("official_source_slice")
    if not isinstance(source, Mapping) or not isinstance(source.get("files"), list):
        raise ActorEpisodeStagingError("preregistration source inventory is missing")
    receipt_files = receipt.get("files")
    if not isinstance(receipt_files, list):
        raise ActorEpisodeStagingError("receipt file inventory is missing")
    by_path = {
        row.get("path"): row for row in receipt_files if isinstance(row, Mapping)
    }
    selected: list[dict[str, Any]] = []
    for body in BODIES:
        for condition in conditions:
            source_condition, episode_count = CONDITION_CONTRACT[condition]
            matches = [
                row
                for row in source["files"]
                if isinstance(row, Mapping)
                and row.get("body") == body
                and row.get("condition") == source_condition
            ]
            if len(matches) != 1:
                raise ActorEpisodeStagingError("source archive identity is missing or ambiguous")
            row = matches[0]
            receipt_row = by_path.get(row.get("path"))
            if (
                not isinstance(receipt_row, Mapping)
                or receipt_row.get("size_bytes") != row.get("size_bytes")
                or receipt_row.get("observed_payload_sha256") != row.get("lfs_sha256")
                or receipt_row.get("payload_sha256_match") is not True
            ):
                raise ActorEpisodeStagingError("receipt/source archive identity differs")
            selected.append(
                {
                    "body": body,
                    "condition": condition,
                    "source_condition": source_condition,
                    "episode_count": episode_count,
                    "path": row["path"],
                    "size_bytes": row["size_bytes"],
                    "payload_sha256": row["lfs_sha256"],
                }
            )
    return selected


def _hash_open_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, HASH_CHUNK_BYTES)
        if not block:
            break
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _open_verified_archive(path: Path, binding: Mapping[str, Any]) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ActorEpisodeStagingError("cannot open selected archive read-only") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != binding["size_bytes"]:
            raise ActorEpisodeStagingError("selected archive size changed after receipt")
        if _hash_open_descriptor(descriptor) != binding["payload_sha256"]:
            raise ActorEpisodeStagingError("selected archive hash changed after receipt")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, before


def _archive_episode_members(
    archive: zipfile.ZipFile,
    binding: Mapping[str, Any],
) -> dict[int, dict[str, zipfile.ZipInfo]]:
    expected_prefix = Path(str(binding["path"])).stem
    expected_count = int(binding["episode_count"])
    episodes: dict[int, dict[str, zipfile.ZipInfo]] = {}
    names: set[str] = set()
    for info in archive.infolist():
        if info.filename in names:
            raise ActorEpisodeStagingError("archive contains duplicate member names")
        names.add(info.filename)
        _safe_relative(info.filename.rstrip("/"), "ZIP member")
        if info.flag_bits & 0x1:
            raise ActorEpisodeStagingError("archive contains encrypted members")
        match = EPISODE_MEMBER.fullmatch(info.filename)
        if match is None:
            continue
        if match.group("prefix") != expected_prefix:
            raise ActorEpisodeStagingError("episode member has an unexpected archive prefix")
        kind = match.group("kind")
        suffix = match.group("suffix")
        expected_kind = {
            "hdf5": "data",
            "json": "instructions",
            "mp4": "video",
            "pkl": "_traj_data",
        }[suffix]
        if kind != expected_kind:
            raise ActorEpisodeStagingError("episode member suffix/directory mismatch")
        episode_id = int(match.group("episode"))
        row = episodes.setdefault(episode_id, {})
        if suffix in row:
            raise ActorEpisodeStagingError("episode contains duplicate typed members")
        row[suffix] = info
    expected_ids = set(range(expected_count))
    if set(episodes) != expected_ids or any(set(row) != {"hdf5", "json", "mp4", "pkl"} for row in episodes.values()):
        raise ActorEpisodeStagingError(
            "archive lacks the exact paired HDF5/JSON/video/pickle episode inventory"
        )
    return episodes


def _normalized_dtype(dtype: Any) -> str:
    if dtype.kind in {"S", "U"}:
        return f"{dtype.kind}:fixed_width_encoded_bytes"
    return str(dtype)


def audit_hdf5_metadata(path: Path, body: str) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        raise ActorEpisodeStagingError("h5py is required for metadata-only HDF5 audit") from error

    datasets: list[dict[str, Any]] = []
    non_hard_links: list[str] = []
    seen_objects: set[int] = set()
    try:
        with h5py.File(path, "r") as handle:
            def walk(group: Any, prefix: str = "") -> None:
                for name in group.keys():
                    full_name = f"{prefix}/{name}" if prefix else name
                    link = group.get(name, getlink=True)
                    if not isinstance(link, h5py.HardLink):
                        non_hard_links.append(full_name)
                        continue
                    obj = group.get(name)
                    address = int(h5py.h5o.get_info(obj.id).addr)
                    if address in seen_objects:
                        continue
                    seen_objects.add(address)
                    if isinstance(obj, h5py.Group):
                        walk(obj, full_name)
                    elif isinstance(obj, h5py.Dataset):
                        if obj.dtype.hasobject or obj.ndim < 1:
                            raise ActorEpisodeStagingError(
                                "HDF5 contains object/vlen dtype or scalar datasets"
                            )
                        datasets.append(
                            {
                                "path": full_name,
                                "shape": list(obj.shape),
                                "dtype": str(obj.dtype),
                                "normalized_dtype": _normalized_dtype(obj.dtype),
                                "compression": obj.compression,
                                "chunks": list(obj.chunks) if obj.chunks is not None else None,
                            }
                        )
                    else:
                        raise ActorEpisodeStagingError("HDF5 contains an unsupported object")

            walk(handle)
    except (OSError, ValueError) as error:
        raise ActorEpisodeStagingError("episode is not a readable HDF5 file") from error
    if non_hard_links:
        raise ActorEpisodeStagingError("HDF5 external or soft links are forbidden")
    by_path = {row["path"]: row for row in datasets}
    if any(name not in by_path for name in REQUIRED_HDF5_DATASETS):
        raise ActorEpisodeStagingError("HDF5 lacks required RoboTwin actor datasets")
    vector = by_path["joint_action/vector"]
    if len(vector["shape"]) != 2 or vector["shape"][0] < 2:
        raise ActorEpisodeStagingError("joint action vector must be [T,D] with T>=2")
    frame_count, action_dim = vector["shape"]
    if action_dim != EXPECTED_ACTION_DIMS[body]:
        raise ActorEpisodeStagingError("body action dimension differs from official contract")
    if any(row["shape"][0] != frame_count for row in datasets):
        raise ActorEpisodeStagingError("HDF5 datasets do not share one frame dimension")
    for name in ("endpose/left_endpose", "endpose/right_endpose"):
        if by_path[name]["shape"] != [frame_count, 7]:
            raise ActorEpisodeStagingError("end-effector pose dataset is not [T,7]")
    for name in ("joint_action/left_arm", "joint_action/right_arm"):
        if len(by_path[name]["shape"]) != 2 or by_path[name]["shape"][1] < 1:
            raise ActorEpisodeStagingError("arm joint dataset is not [T,D]")
    for name in (
        "endpose/left_gripper",
        "endpose/right_gripper",
        "joint_action/left_gripper",
        "joint_action/right_gripper",
    ):
        if by_path[name]["shape"] != [frame_count]:
            raise ActorEpisodeStagingError("gripper dataset is not [T]")
    component_dim = (
        by_path["joint_action/left_arm"]["shape"][1]
        + 1
        + by_path["joint_action/right_arm"]["shape"][1]
        + 1
    )
    if component_dim != action_dim:
        raise ActorEpisodeStagingError("joint action components do not match vector dimension")
    structural = [
        {
            "path": row["path"],
            "rank": len(row["shape"]),
            "tail_shape": row["shape"][1:],
            "normalized_dtype": row["normalized_dtype"],
        }
        for row in sorted(datasets, key=lambda item: item["path"])
    ]
    return {
        "metadata_only": True,
        "dataset_values_read": False,
        "external_or_soft_links_present": False,
        "dataset_count": len(datasets),
        "frame_count": frame_count,
        "action_dim": action_dim,
        "dataset_inventory_sha256": canonical_sha256(datasets),
        "structural_schema_sha256": canonical_sha256(structural),
        "required_actor_datasets_present": True,
    }


def validate_instruction_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActorEpisodeStagingError("instruction member is not UTF-8 JSON") from error
    if not isinstance(value, dict) or set(value) != {"seen", "unseen"}:
        raise ActorEpisodeStagingError("instruction JSON must contain only seen/unseen lists")
    for split in ("seen", "unseen"):
        rows = value[split]
        if (
            not isinstance(rows, list)
            or len(rows) != 100
            or any(not isinstance(item, str) or not item.strip() for item in rows)
        ):
            raise ActorEpisodeStagingError("instruction split must contain 100 nonempty strings")
    return {
        "schema": "robotwin2_seen_unseen_instruction_lists_v1",
        "seen_count": 100,
        "unseen_count": 100,
        "instruction_text_selected": False,
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _stream_member_to_temp(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    work_dir: Path,
) -> tuple[Path, str]:
    if not 0 < info.file_size <= MAX_HDF5_BYTES:
        raise ActorEpisodeStagingError("HDF5 member size is outside the bounded contract")
    descriptor, name = tempfile.mkstemp(
        prefix="episode.", suffix=".hdf5.partial", dir=work_dir
    )
    path = Path(name)
    digest = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as target, archive.open(info, "r") as source:
            while True:
                block = source.read(HASH_CHUNK_BYTES)
                if not block:
                    break
                total += len(block)
                if total > info.file_size:
                    raise ActorEpisodeStagingError("HDF5 member exceeded declared ZIP size")
                digest.update(block)
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if total != info.file_size:
            raise ActorEpisodeStagingError("HDF5 member did not match declared ZIP size")
        return path, digest.hexdigest()
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_instruction_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if not 0 < info.file_size <= MAX_JSON_BYTES:
        raise ActorEpisodeStagingError("instruction JSON size is outside the bounded contract")
    with archive.open(info, "r") as source:
        raw = source.read(MAX_JSON_BYTES + 1)
    if len(raw) != info.file_size or len(raw) > MAX_JSON_BYTES:
        raise ActorEpisodeStagingError("instruction JSON did not match declared ZIP size")
    return raw


def _publish_file_from_temp(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    source.chmod(0o444)
    os.link(source, target)
    source.unlink()


def _publish_bytes_new(target: Path, raw: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o444)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def stage_archive(
    archive_path: Path,
    binding: Mapping[str, Any],
    build_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    descriptor, before = _open_verified_archive(archive_path, binding)
    episode_rows: list[dict[str, Any]] = []
    schemas: set[str] = set()
    try:
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            try:
                archive = zipfile.ZipFile(handle, "r")
            except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
                raise ActorEpisodeStagingError("selected payload is not a readable ZIP") from error
            with archive:
                episodes = _archive_episode_members(archive, binding)
                env_name = f"{binding['body']}_{binding['condition']}"
                relative_base = Path("data") / "RoboTwin2_move_can_pot" / TASK / env_name
                for episode_id in sorted(episodes):
                    members = episodes[episode_id]
                    temporary, hdf_sha = _stream_member_to_temp(
                        archive, members["hdf5"], work_dir
                    )
                    try:
                        hdf_audit = audit_hdf5_metadata(temporary, str(binding["body"]))
                        relative_hdf = relative_base / "data" / f"episode{episode_id}.hdf5"
                        _publish_file_from_temp(temporary, build_root / relative_hdf)
                    finally:
                        temporary.unlink(missing_ok=True)
                    instruction_raw = _read_instruction_member(archive, members["json"])
                    instruction_audit = validate_instruction_json(instruction_raw)
                    relative_instruction = (
                        relative_base / "instructions" / f"episode{episode_id}.json"
                    )
                    _publish_bytes_new(build_root / relative_instruction, instruction_raw)
                    schemas.add(hdf_audit["structural_schema_sha256"])
                    episode_rows.append(
                        {
                            "episode_id": episode_id,
                            "hdf5_path": relative_hdf.as_posix(),
                            "instruction_json_path": relative_instruction.as_posix(),
                            "source_hdf5_member": members["hdf5"].filename,
                            "source_instruction_member": members["json"].filename,
                            "source_video_member_verified_not_opened": members["mp4"].filename,
                            "source_pickle_member_verified_not_opened": members["pkl"].filename,
                            "hdf5_size_bytes": members["hdf5"].file_size,
                            "hdf5_sha256": hdf_sha,
                            "instruction_json_size_bytes": len(instruction_raw),
                            "instruction_json_sha256": instruction_audit["content_sha256"],
                            "hdf5_audit": hdf_audit,
                            "instruction_audit": instruction_audit,
                        }
                    )
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise ActorEpisodeStagingError("selected archive changed during staging")
    finally:
        os.close(descriptor)
    if len(schemas) != 1:
        raise ActorEpisodeStagingError("within-archive HDF5 structural schema drift detected")
    return {
        **dict(binding),
        "archive_path_used": str(archive_path),
        "archive_rehashed_before_member_access": True,
        "archive_payload_sha256_match_after_receipt": True,
        "episode_count_staged": len(episode_rows),
        "frame_count_total": sum(row["hdf5_audit"]["frame_count"] for row in episode_rows),
        "action_dim": EXPECTED_ACTION_DIMS[str(binding["body"])],
        "structural_schema_sha256": next(iter(schemas)),
        "episodes": episode_rows,
    }


def validate_manifest(value: Mapping[str, Any]) -> None:
    document = dict(value)
    digest = document.pop("manifest_sha256", None)
    archives = document.get("archives")
    boundary = document.get("read_boundary")
    authority = document.get("authority")
    adapter = document.get("canonical_adapter_interface")
    if not _is_sha256(digest) or digest != canonical_sha256(document):
        raise ActorEpisodeStagingError("staging manifest canonical SHA-256 changed")
    if (
        document.get("format") != FORMAT
        or document.get("status") != STATUS
        or document.get("task") != TASK
        or document.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or not _is_sha256(document.get("materialization_receipt_sha256"))
        or not _is_sha256(document.get("preregistration_file_sha256"))
        or not _is_sha256(document.get("materialization_receipt_file_sha256"))
        or not isinstance(archives, list)
        or not archives
        or document.get("archive_count") != len(archives)
        or document.get("episode_count")
        != sum(row.get("episode_count_staged", -1) for row in archives)
        or not isinstance(boundary, Mapping)
        or boundary.get("pickle_members_opened") != 0
        or boundary.get("video_members_opened") != 0
        or boundary.get("hdf5_dataset_values_read") is not False
        or boundary.get("bulk_archive_extraction_used") is not False
        or not isinstance(adapter, Mapping)
        or adapter.get("format") != canonical_adapter.FORMAT
        or adapter.get("logical_sha256")
        != canonical_adapter.contract()["logical_sha256"]
        or adapter.get("implementation_file") != Path(canonical_adapter.__file__).name
        or adapter.get("implementation_file_sha256")
        != file_sha256(Path(canonical_adapter.__file__).resolve())
        or adapter.get("action_effect14_materialized_by_this_staging_run") is not False
        or adapter.get("state27_materialized_by_this_staging_run") is not False
        or not isinstance(authority, Mapping)
        or authority.get("actor_training_data_staging_complete") is not True
        or authority.get("actor_training_authorized") is not False
        or authority.get("critic_or_shared_event_head_training_authorized") is not False
        or authority.get("success_failure_recovery_object_event_supervision_generated") is not False
        or authority.get("task_success_or_cross_embodiment_claim_authorized") is not False
    ):
        raise ActorEpisodeStagingError("staging manifest violates the data-only boundary")


def _output_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if absolute.name in {"", ".", ".."} or any(parent.is_symlink() for parent in absolute.parents):
        raise ActorEpisodeStagingError("output path is unsafe or has a symlink parent")
    parent = absolute.parent.resolve(strict=True)
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(absolute)
    if not parent.is_dir():
        raise ActorEpisodeStagingError("output parent is not a directory")
    return absolute


def _write_manifest(build_root: Path, value: Mapping[str, Any]) -> None:
    validate_manifest(value)
    target = build_root / "actor_staging_manifest.json"
    raw = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _publish_bytes_new(target, raw)


def _freeze_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, child_dirs, files in os.walk(root):
        directory = Path(current)
        directories.append(directory)
        for filename in files:
            path = directory / filename
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ActorEpisodeStagingError("staging tree contains a non-regular file")
            path.chmod(0o444)
        for child in child_dirs:
            child_path = directory / child
            if child_path.is_symlink():
                raise ActorEpisodeStagingError("staging tree contains a symlink directory")
    for directory in reversed(directories):
        directory.chmod(0o555)


def build_staging(
    preregistration: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_file_binding: Mapping[str, Any],
    download_root: Path,
    output: Path,
    conditions: Sequence[str],
) -> dict[str, Any]:
    root = _real_directory(download_root, "download root")
    selected = select_archive_bindings(preregistration, receipt, conditions)
    output = _output_path(output)
    build_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=output.parent)
    )
    work_dir = build_root / ".episode_work"
    work_dir.mkdir(mode=0o700)
    try:
        archive_rows = [
            stage_archive(
                _contained_regular(root, str(binding["path"]), "selected archive"),
                binding,
                build_root,
                work_dir,
            )
            for binding in selected
        ]
        work_dir.rmdir()
        unsigned: dict[str, Any] = {
            "format": FORMAT,
            "status": STATUS,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "bodies": list(BODIES),
            "conditions": list(conditions),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "materialization_receipt_sha256": receipt.get(
                "materialization_receipt_sha256"
            ),
            **dict(source_file_binding),
            "receipt_recorded_download_root": receipt.get("download_root"),
            "download_root_used": str(root),
            "download_root_may_be_relocated_because_payload_identity_is_reverified": True,
            "staging_root": str(output),
            "layout": {
                "kind": "xpolicylab_discovery_shaped_raw_robotwin2_legacy_hdf5_v1",
                "pattern": (
                    "data/RoboTwin2_move_can_pot/move_can_pot/"
                    "<body>_<condition>/{data,instructions}/episodeN.*"
                ),
                "generic_xpolicylab_lerobot_v3_direct_converter_ready": False,
                "reason": (
                    "official archives use legacy observation/joint_action/endpose HDF5; "
                    "the generic converter expects XPolicyLab vision/state/action v1.0"
                ),
                "followup_label_free_schema_adapter_required": True,
            },
            "canonical_adapter_interface": {
                **canonical_adapter.contract(),
                "implementation_file": Path(canonical_adapter.__file__).name,
                "implementation_file_sha256": file_sha256(
                    Path(canonical_adapter.__file__).resolve()
                ),
                "action_effect14_materialized_by_this_staging_run": False,
                "state27_materialized_by_this_staging_run": False,
                "reason": (
                    "staging preserves raw expert episodes; a later explicit adapter may read "
                    "endpose/gripper arrays, while state27 additionally needs object/goal/event/"
                    "predicate inputs unavailable from this expert-only staging contract"
                ),
            },
            "archive_count": len(archive_rows),
            "episode_count": sum(row["episode_count_staged"] for row in archive_rows),
            "frame_count_total": sum(row["frame_count_total"] for row in archive_rows),
            "archives": archive_rows,
            "read_boundary": {
                "selected_archive_payloads_rehashed": True,
                "one_hdf5_member_at_a_time_streamed_to_controlled_temporary_file": True,
                "hdf5_metadata_read": True,
                "hdf5_dataset_values_read": False,
                "instruction_json_parsed": True,
                "pickle_members_opened": 0,
                "pickle_members_deserialized": 0,
                "video_members_opened": 0,
                "video_or_image_frames_decoded": 0,
                "bulk_archive_extraction_used": False,
            },
            "supervision_semantics": {
                "public_expert_demonstrations_only": True,
                "success_labels_generated": False,
                "failure_labels_generated": False,
                "recovery_labels_generated": False,
                "object_change_labels_generated": False,
                "event_labels_generated": False,
                "expert_trajectory_assumed_to_prove_task_success": False,
            },
            "authority": {
                "actor_training_data_staging_complete": True,
                "actor_training_authorized": False,
                "critic_or_shared_event_head_training_authorized": False,
                "success_failure_recovery_object_event_supervision_generated": False,
                "simulator_execution_authorized": False,
                "evaluation_or_checkpoint_promotion_authorized": False,
                "task_success_or_cross_embodiment_claim_authorized": False,
            },
            "empirical_result": None,
        }
        manifest = {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}
        _write_manifest(build_root, manifest)
        _freeze_tree(build_root)
        os.rename(build_root, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    except Exception:
        if build_root.exists():
            for current, directories, files in os.walk(build_root):
                Path(current).chmod(0o700)
                for filename in files:
                    (Path(current) / filename).chmod(0o600)
                for directory in directories:
                    (Path(current) / directory).chmod(0o700)
            shutil.rmtree(build_root)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument(
        "--download-root",
        type=Path,
        help="Optional relocated snapshot root; defaults to the receipt-recorded root.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_CONTRACT),
        default=["clean"],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preregistration, receipt, file_binding = load_source_contracts(
        args.preregistration, args.materialization_receipt
    )
    download_root = (
        args.download_root
        if args.download_root is not None
        else Path(str(receipt.get("download_root", "")))
    )
    manifest = build_staging(
        preregistration,
        receipt,
        file_binding,
        download_root,
        args.output,
        args.conditions,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
