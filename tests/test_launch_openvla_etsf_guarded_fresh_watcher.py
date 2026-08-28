from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_guarded_fresh_watcher as watcher  # noqa: E402


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _args(tmp_path: Path) -> argparse.Namespace:
    for name in ("factual", "model", "rlinf", "robotwin", "robotwin_code"):
        (tmp_path / name).mkdir()
    event = tmp_path / "event.json"
    event.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        upstream_state=tmp_path / "upstream.json",
        counterfactual_root=tmp_path / "oof" / "final",
        fresh_seed_manifest=tmp_path / "fresh.json",
        data=tmp_path / "merged250",
        factual_root=tmp_path / "factual",
        event_spec=event,
        model_path=tmp_path / "model",
        rlinf_root=tmp_path / "rlinf",
        robotwin_root=tmp_path / "robotwin",
        robotwin_code=tmp_path / "robotwin_code",
        output=tmp_path / "fresh_output",
        state_root=tmp_path / "watcher",
        python_bin=Path(sys.executable),
        fresh_launcher=SCRIPTS / "launch_openvla_etsf_fresh50_confirmation.py",
        wait_timeout_seconds=1.0,
        child_wait_timeout_seconds=1.0,
        poll_seconds=0.01,
    )


def test_unauthorized_upstream_exits_without_reading_or_launching_fresh(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    _json(
        args.upstream_state,
        {
            "status": watcher.FORBIDDEN,
            "oof_output": str(tmp_path / "oof"),
        },
    )
    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    result = watcher.execute(args)
    assert result["status"].endswith("fresh_forbidden")
    assert result["fresh_manifest_read_by_watcher"] is False
    assert not args.fresh_seed_manifest.exists()
    assert not args.output.exists()


def test_authorized_upstream_launches_exactly_one_child_and_checks_marker(
    monkeypatch, tmp_path: Path
) -> None:
    args = _args(tmp_path)
    _json(
        args.upstream_state,
        {
            "status": watcher.READY,
            "oof_output": str(tmp_path / "oof"),
        },
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        _json(args.output / "pipeline_audit.json", {"status": "complete"})
        _json(
            args.output / "fresh50_evaluation" / "evaluated_once.json",
            {"status": "complete"},
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(watcher.subprocess, "run", fake_run)
    result = watcher.execute(args)
    assert len(calls) == 1
    assert "--fresh-seed-manifest" in calls[0]
    assert calls[0][calls[0].index("--counterfactual-root") + 1] == str(
        args.counterfactual_root.absolute()
    )
    assert result["status"] == "complete_fresh50_confirmed"
    assert result["fresh_manifest_read_by_watcher"] is False
