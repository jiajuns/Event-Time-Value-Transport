from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_schema6_post_collection as pipeline  # noqa: E402


def _signed_authority(request: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "format": pipeline.INDEPENDENT_AUTHORITY_FORMAT,
        "status": "authorized_for_external_paired400_launcher_only",
        "request_sha256": request["request_sha256"],
        "calibration_receipt_sha256": request["calibration_receipt_sha256"],
        "authorized_pair_count": 400,
        "execution_authorized": True,
        "external_paired_launcher_required": True,
        "post_collection_orchestrator_may_execute_pairs": False,
        "sealed_identity_payload_disclosed": False,
        "sealed_outcome_or_label_payload_disclosed": False,
        "target_validation50_hdf5_open_authorized": False,
        "evaluation400_hdf5_or_label_open_authorized": False,
    }
    value["authority_sha256"] = pipeline.canonical_sha256(value)
    return value


def test_gpu_wait_requires_two_idle_audits_after_busy() -> None:
    replies = iter(
        [
            "NVIDIA GeForce RTX 4090, GPU-A\n",
            "91\n",
            "NVIDIA GeForce RTX 4090, GPU-A\n",
            "\n",
            "NVIDIA GeForce RTX 4090, GPU-A\n",
            "\n",
        ]
    )
    slept: list[float] = []
    result = pipeline.wait_two_idle(
        0,
        interval=0.25,
        run_text=lambda _command: next(replies),
        sleep=slept.append,
    )
    assert len(result) == 2
    assert all(not row["compute_pids"] for row in result)
    assert len(slept) == 2


def test_collection_wait_uses_injected_terminal_validator_without_hdf_access(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collection_root"
    (root / "_watcher").mkdir(parents=True)
    (root / "_watcher" / "final_receipt.json").write_text("{}")
    calls: list[Path] = []

    def validate(path: Path) -> dict[str, object]:
        calls.append(path)
        return {"authenticated": True, "hdf5_files_opened": 0}

    result = pipeline.wait_for_collection_terminal(
        root,
        interval=0.1,
        validate=validate,
        sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )
    assert result == {"authenticated": True, "hdf5_files_opened": 0}
    assert calls == [root.resolve()]


def test_independent_authority_wait_is_metadata_only_and_exact(tmp_path: Path) -> None:
    request = {
        "request_sha256": "1" * 64,
        "calibration_receipt_sha256": "2" * 64,
    }
    path = tmp_path / "paired400_authority.json"
    path.write_text(json.dumps(_signed_authority(request)), encoding="utf-8")
    path.chmod(0o444)
    result = pipeline.wait_for_independent_authority(
        path,
        request=request,
        interval=0.1,
        sleep=lambda _seconds: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )
    assert result["authorized_pair_count"] == 400
    assert result["post_collection_orchestrator_may_execute_pairs"] is False

    path.chmod(0o644)
    tampered = _signed_authority(request)
    tampered["sealed_labels_path"] = "/forbidden/labels.npz"
    tampered["authority_sha256"] = pipeline.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "authority_sha256"}
    )
    path.write_text(json.dumps(tampered), encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(pipeline.PostCollectionError, match="scope changed"):
        pipeline.validate_independent_authority(path, request=request)


def test_internal20_formal_calibration_is_categorically_rejected(tmp_path: Path) -> None:
    with pytest.raises(pipeline.PostCollectionError, match="cannot satisfy"):
        pipeline._reject_internal20_formal_calibration_authority(tmp_path, [])
