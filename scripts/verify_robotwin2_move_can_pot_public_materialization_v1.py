#!/usr/bin/env python3
"""Read-only verifier for the public RoboTwin2 ``move_can_pot`` slice.

The verifier hashes the exact archives frozen by the five-body LOBO
preregistration and reads ZIP central-directory metadata only.  It never
extracts an archive and never opens or deserializes a member payload.
Successful verification creates one immutable JSON receipt.  A failed audit
raises before any receipt is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg


FORMAT = "etsf_robotwin2_move_can_pot_public_materialization_receipt_v1"
STATUS = "verified_complete_public_payload_materialization_no_operational_authority"
HASH_CHUNK_BYTES = 8 * 1024 * 1024
MAX_PREREGISTRATION_BYTES = 8 * 1024 * 1024


class PublicMaterializationError(RuntimeError):
    """The public download is incomplete, changed, unsafe, or ambiguous."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_existing_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        metadata = absolute.lstat()
    except FileNotFoundError as exc:
        raise PublicMaterializationError(f"{label} does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PublicMaterializationError(f"{label} must be a real directory, not a symlink")
    return absolute


def _checked_scope(root: Path, task_path: str) -> Path:
    pure = PurePosixPath(task_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise PublicMaterializationError("preregistered task path is not a safe relative path")
    current = root
    for part in pure.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise PublicMaterializationError(f"missing official task directory: {task_path}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise PublicMaterializationError("official task path contains a symlink or non-directory")
    return current


def _scope_inventory(root: Path, scope: Path) -> list[str]:
    files: list[str] = []
    pending = [scope]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                path = Path(entry.path)
                if stat.S_ISLNK(metadata.st_mode):
                    raise PublicMaterializationError("official task scope contains a symbolic link")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    files.append(path.relative_to(root).as_posix())
                else:
                    raise PublicMaterializationError("official task scope contains a special file")
    return sorted(files)


def _safe_member_name(name: str) -> bool:
    if not name or "\\" in name or "\x00" in name:
        return False
    pure = PurePosixPath(name)
    return (
        not pure.is_absolute()
        and bool(pure.parts)
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def _safe_official_path(path: str, task_path: str) -> bool:
    if not path or "\\" in path or "\x00" in path:
        return False
    pure = PurePosixPath(path)
    task = PurePosixPath(task_path)
    return (
        not pure.is_absolute()
        and all(part not in {"", ".", ".."} for part in pure.parts)
        and len(pure.parts) == len(task.parts) + 1
        and pure.parts[: len(task.parts)] == task.parts
    )


def _central_directory_audit(handle: BinaryIO) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(handle, mode="r")
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise PublicMaterializationError("official payload is not a readable ZIP archive") from exc

    digest = hashlib.sha256()
    names: set[str] = set()
    compression_methods: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    member_count = 0
    directory_count = 0
    total_compressed = 0
    total_uncompressed = 0
    maximum_uncompressed = 0
    encrypted_count = 0
    unsafe_count = 0
    duplicate_count = 0
    special_type_count = 0
    try:
        for ordinal, info in enumerate(archive.infolist()):
            member_count += 1
            is_directory = info.is_dir()
            directory_count += int(is_directory)
            total_compressed += info.compress_size
            total_uncompressed += info.file_size
            maximum_uncompressed = max(maximum_uncompressed, info.file_size)
            compression_methods[str(info.compress_type)] += 1
            suffix = PurePosixPath(info.filename.rstrip("/")).suffix.lower() or "<none>"
            suffixes[suffix] += int(not is_directory)
            encrypted_count += int(bool(info.flag_bits & 0x1))
            unsafe_count += int(not _safe_member_name(info.filename))
            duplicate_count += int(info.filename in names)
            names.add(info.filename)

            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if unix_mode and stat.S_IFMT(unix_mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                special_type_count += 1

            row = {
                "ordinal": ordinal,
                "name": info.filename,
                "crc32": info.CRC,
                "compression_method": info.compress_type,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "flag_bits": info.flag_bits,
                "external_attr": info.external_attr,
                "is_directory": is_directory,
            }
            digest.update(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")
        comment_sha256 = hashlib.sha256(archive.comment).hexdigest()
    finally:
        archive.close()

    if member_count == 0:
        raise PublicMaterializationError("official ZIP archive has no central-directory members")
    if unsafe_count or duplicate_count or encrypted_count or special_type_count:
        raise PublicMaterializationError(
            "ZIP central directory contains unsafe, duplicate, encrypted, or special members"
        )
    return {
        "central_directory_read_only": True,
        "member_payload_bytes_read": 0,
        "member_count": member_count,
        "file_member_count": member_count - directory_count,
        "directory_member_count": directory_count,
        "central_directory_inventory_sha256": digest.hexdigest(),
        "member_paths_safe_relative_posix": True,
        "duplicate_member_name_count": duplicate_count,
        "encrypted_member_count": encrypted_count,
        "special_file_type_member_count": special_type_count,
        "compression_method_member_counts": dict(sorted(compression_methods.items())),
        "file_suffix_member_counts": dict(sorted(suffixes.items())),
        "total_member_compressed_bytes_declared": total_compressed,
        "total_member_uncompressed_bytes_declared": total_uncompressed,
        "maximum_member_uncompressed_bytes_declared": maximum_uncompressed,
        "archive_comment_sha256": comment_sha256,
        "crc_values_are_central_directory_metadata_not_payload_recomputation": True,
        "member_payload_crc_verified": False,
    }


def _hash_and_audit_zip(path: Path, expected_size: int) -> tuple[str, dict[str, Any]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PublicMaterializationError(f"cannot open official archive read-only: {path.name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise PublicMaterializationError(f"official archive size mismatch: {path.name}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            zip_audit = _central_directory_audit(handle)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise PublicMaterializationError(f"official archive changed during audit: {path.name}")
        return digest.hexdigest(), zip_audit
    finally:
        os.close(descriptor)


def _read_preregistration_file(path: Path) -> Mapping[str, Any]:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise PublicMaterializationError("cannot open preregistration as a regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PREREGISTRATION_BYTES:
            raise PublicMaterializationError("preregistration is not a bounded regular file")
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8", closefd=True) as handle:
            value = json.load(handle)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicMaterializationError("preregistration is not valid UTF-8 JSON") from exc
    finally:
        os.close(descriptor)
    if not isinstance(value, Mapping):
        raise PublicMaterializationError("preregistration JSON must be an object")
    return value


def load_preregistration(path: Path | None) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    value = prereg.build_preregistration() if path is None else _read_preregistration_file(path)
    try:
        audit = prereg.validate_preregistration(value)
    except prereg.CrossEmbodimentSlicePreregistrationError as exc:
        raise PublicMaterializationError("preregistration is not the exact reviewed contract") from exc
    return value, audit, "deterministic_module_rebuild" if path is None else "provided_json"


def build_receipt(
    download_root: Path,
    preregistration: Mapping[str, Any],
    preregistration_audit: Mapping[str, Any],
    preregistration_source: str,
) -> dict[str, Any]:
    root = _absolute_existing_directory(download_root, "download root")
    source = preregistration.get("official_source_slice")
    if not isinstance(source, Mapping):
        raise PublicMaterializationError("preregistration lacks official source slice")
    task_path = source.get("task_path")
    files = source.get("files")
    if not isinstance(task_path, str) or not isinstance(files, list) or not files:
        raise PublicMaterializationError("preregistered official inventory is malformed")
    scope = _checked_scope(root, task_path)

    expected_paths: list[str] = []
    for row in files:
        if not isinstance(row, Mapping):
            raise PublicMaterializationError("preregistered official file row is malformed")
        path = row.get("path")
        size = row.get("size_bytes")
        sha256 = row.get("lfs_sha256")
        if (
            not isinstance(path, str)
            or type(size) is not int
            or size <= 0
            or not _is_sha256(sha256)
            or not _safe_official_path(path, task_path)
        ):
            raise PublicMaterializationError("preregistered path, size, or payload hash is malformed")
        expected_paths.append(path)
    if len(expected_paths) != len(set(expected_paths)):
        raise PublicMaterializationError("preregistered official inventory contains duplicates")

    observed_paths = _scope_inventory(root, scope)
    if observed_paths != sorted(expected_paths):
        missing = sorted(set(expected_paths) - set(observed_paths))
        extra = sorted(set(observed_paths) - set(expected_paths))
        raise PublicMaterializationError(
            f"official task inventory mismatch: missing={missing!r}, extra={extra!r}"
        )

    verified_files: list[dict[str, Any]] = []
    for row in files:
        relative = str(row["path"])
        expected_size = int(row["size_bytes"])
        expected_sha256 = str(row["lfs_sha256"])
        observed_sha256, zip_audit = _hash_and_audit_zip(root / relative, expected_size)
        if observed_sha256 != expected_sha256:
            raise PublicMaterializationError(f"official archive SHA-256 mismatch: {relative}")
        verified_files.append(
            {
                "path": relative,
                "size_bytes": expected_size,
                "expected_payload_sha256": expected_sha256,
                "observed_payload_sha256": observed_sha256,
                "size_match": True,
                "payload_sha256_match": True,
                "zip_central_directory_audit": zip_audit,
            }
        )

    expected_total = sum(int(row["size_bytes"]) for row in files)
    if expected_total != source.get("total_size_bytes"):
        raise PublicMaterializationError("preregistered total size does not equal official file rows")
    unsigned: dict[str, Any] = {
        "format": FORMAT,
        "status": STATUS,
        "materialized": True,
        "materialized_definition": (
            "exact_official_task_file_set_and_every_size_and_archive_payload_sha256_verified"
        ),
        "download_root": str(root),
        "official_task_scope": task_path,
        "preregistration_source": preregistration_source,
        "preregistration_sha256": preregistration.get("preregistration_sha256"),
        "hf_repo_id": source.get("hf_repo_id"),
        "hf_repo_revision": source.get("hf_repo_revision"),
        "official_file_count": len(verified_files),
        "official_total_size_bytes": expected_total,
        "no_missing_or_extra_official_task_files": True,
        "all_exact_sizes_verified": True,
        "all_exact_archive_payload_sha256_verified": True,
        "all_zip_central_directories_audited": True,
        "files": verified_files,
        "preregistration_validation": dict(preregistration_audit),
        "implementation_binding": {
            "verifier_module": Path(__file__).name,
            "verifier_file_sha256": _file_sha256(Path(__file__).resolve()),
            "preregistration_module": Path(prereg.__file__).name,
            "preregistration_module_file_sha256": _file_sha256(
                Path(prereg.__file__).resolve()
            ),
        },
        "read_boundary": {
            "archive_bytes_read_only_for_payload_sha256": True,
            "zip_central_directory_metadata_read": True,
            "zip_member_payload_bytes_read": 0,
            "archive_extracted": False,
            "pickle_payload_opened_or_deserialized": False,
            "numpy_payload_opened_or_deserialized": False,
            "torch_payload_opened_or_deserialized": False,
            "video_or_image_member_decoded": False,
            "dataset_semantics_or_labels_validated": False,
        },
        "authority": {
            "download_completeness_attested": True,
            "training_authorized": False,
            "evaluation_authorized": False,
            "simulator_execution_authorized": False,
            "checkpoint_selection_or_promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_performance_claim_authorized": False,
        },
        "empirical_result": None,
    }
    return {**unsigned, "materialization_receipt_sha256": canonical_sha256(unsigned)}


def validate_receipt(value: Mapping[str, Any]) -> None:
    document = dict(value)
    digest = document.pop("materialization_receipt_sha256", None)
    files = document.get("files")
    authority = document.get("authority")
    read_boundary = document.get("read_boundary")
    implementation = document.get("implementation_binding")
    if not _is_sha256(digest) or digest != canonical_sha256(document):
        raise PublicMaterializationError("materialization receipt canonical SHA changed")
    if (
        document.get("format") != FORMAT
        or document.get("status") != STATUS
        or document.get("materialized") is not True
        or document.get("no_missing_or_extra_official_task_files") is not True
        or document.get("all_exact_sizes_verified") is not True
        or document.get("all_exact_archive_payload_sha256_verified") is not True
        or not isinstance(files, list)
        or not files
        or any(not isinstance(row, Mapping) for row in files)
        or document.get("official_file_count") != len(files)
        or document.get("official_total_size_bytes")
        != sum(row.get("size_bytes", -1) for row in files)
        or any(
            row.get("size_match") is not True
            or row.get("payload_sha256_match") is not True
            or row.get("expected_payload_sha256") != row.get("observed_payload_sha256")
            or not _is_sha256(row.get("observed_payload_sha256"))
            or not isinstance(row.get("zip_central_directory_audit"), Mapping)
            or row["zip_central_directory_audit"].get("central_directory_read_only") is not True
            or row["zip_central_directory_audit"].get("member_payload_bytes_read") != 0
            for row in files
        )
        or not isinstance(read_boundary, Mapping)
        or read_boundary.get("zip_member_payload_bytes_read") != 0
        or read_boundary.get("archive_extracted") is not False
        or read_boundary.get("pickle_payload_opened_or_deserialized") is not False
        or read_boundary.get("numpy_payload_opened_or_deserialized") is not False
        or read_boundary.get("torch_payload_opened_or_deserialized") is not False
        or not isinstance(implementation, Mapping)
        or implementation.get("verifier_module") != Path(__file__).name
        or implementation.get("preregistration_module") != Path(prereg.__file__).name
        or implementation.get("verifier_file_sha256")
        != _file_sha256(Path(__file__).resolve())
        or implementation.get("preregistration_module_file_sha256")
        != _file_sha256(Path(prereg.__file__).resolve())
        or not isinstance(authority, Mapping)
        or authority
        != {
            "download_completeness_attested": True,
            "training_authorized": False,
            "evaluation_authorized": False,
            "simulator_execution_authorized": False,
            "checkpoint_selection_or_promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_performance_claim_authorized": False,
        }
    ):
        raise PublicMaterializationError("receipt does not prove complete non-authorizing materialization")


def _output_path(value: Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if any(parent.is_symlink() for parent in output.parents):
        raise PublicMaterializationError("output path contains a symbolic-link parent")
    output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    return output


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    validate_receipt(value)
    output = _output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        help="Exact create-once LOBO preregistration JSON; default rebuilds reviewed module contract.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preregistration, audit, source = load_preregistration(args.preregistration)
    receipt = build_receipt(args.download_root, preregistration, audit, source)
    write_json_new(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
