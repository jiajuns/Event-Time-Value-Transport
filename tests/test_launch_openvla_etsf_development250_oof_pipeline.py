from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_development250_oof_pipeline as pipeline  # noqa: E402
from audit_openvla_etsf_development250 import FORMAT as AUDIT_FORMAT  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    old = tmp_path / "old100"
    new = tmp_path / "new150"
    old.mkdir()
    new.mkdir()
    event_spec = tmp_path / "event_spec.json"
    pretrained = tmp_path / "factual.pt"
    event_spec.write_text("{}", encoding="utf-8")
    pretrained.write_bytes(b"factual")
    return argparse.Namespace(
        audit=tmp_path / "audit.json",
        old_development100=old,
        new_development150=new,
        merged_output=tmp_path / "merged250",
        oof_output=tmp_path / "oof250",
        state_root=tmp_path / "pipeline",
        pretrained=pretrained,
        event_spec=event_spec,
        python_bin=Path(sys.executable),
        merge_script=SCRIPTS / "merge_openvla_etsf_schema5_development.py",
        oof_launcher=SCRIPTS / "launch_openvla_etsf_counterfactual_oof_v5.py",
        oof_trainer=SCRIPTS / "train_openvla_etsf_counterfactual_oof.py",
        gpu_index=0,
        num_workers=0,
        wait_timeout_seconds=1.0,
        poll_seconds=0.01,
    )


def _audit(args: argparse.Namespace) -> dict:
    value = {
        "format": AUDIT_FORMAT,
        "status": "training_ready",
        "training_authorized": True,
        "event_spec": {
            "path": str(args.event_spec.resolve()),
            "sha256": pipeline.sha256(args.event_spec),
        },
        "old100": {
            "root": str(args.old_development100.resolve()),
            "groups": 100,
            "candidate_count": 4,
        },
        "development150": {
            "root": str(args.new_development150.resolve()),
            "groups": 150,
            "candidate_count": 5,
        },
        "combined": {"groups": 250, "candidate_branches": 1150},
        "fresh50_exclusion": {"labels_read": False},
    }
    value["audit_payload_sha256"] = pipeline.canonical_sha256(value)
    _write_json(args.audit, value)
    return value


def test_training_ready_audit_is_signed_and_source_bound(tmp_path: Path) -> None:
    args = _args(tmp_path)
    value = _audit(args)
    result = pipeline.validate_training_ready_audit(args)
    assert result["groups"] == 250
    assert result["candidate_branches"] == 1150

    value["combined"]["groups"] = 249
    _write_json(args.audit, value)
    with pytest.raises(RuntimeError, match="signature changed"):
        pipeline.validate_training_ready_audit(args)


def test_pipeline_runs_merge_then_oof_and_never_opens_fresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    _audit(args)
    calls = []

    def fake_run(command, *, log, environment):
        del environment
        calls.append((log.name, list(command)))
        if log.name == "merge.log":
            args.merged_output.mkdir()
            _write_json(
                args.merged_output / "manifest.json",
                {
                    "status": "complete",
                    "completed": 250,
                    "seed_registry": (
                        "merged_official100_plus_explicit_development150"
                    ),
                    "fresh_confirmation_labels_read": False,
                },
            )
        else:
            args.oof_output.mkdir()
            _write_json(
                args.oof_output / "launch_state.json",
                {"status": "stopped_guard_not_authorized"},
            )
        return {"returncode": 0, "log": str(log), "argv": list(command)}

    monkeypatch.setattr(pipeline, "run_logged", fake_run)
    monkeypatch.setattr(
        pipeline,
        "wait_for_idle_4090",
        lambda *args, **kwargs: {
            "gpu_index": 0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "compute_pids": [],
        },
    )
    result = pipeline.execute(args)
    assert [name for name, _ in calls] == ["merge.log", "oof.log"]
    assert result["status"] == "complete_guard_not_authorized_fresh_forbidden"
    assert result["fresh_confirmation_policy"] == "forbidden"
    assert all("fresh" not in " ".join(command).lower() for _, command in calls)
