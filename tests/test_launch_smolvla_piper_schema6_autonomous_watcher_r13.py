from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for path in (SCRIPTS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import launch_smolvla_piper_schema6_autonomous_watcher_r13 as watcher  # noqa: E402

legacy_fixtures = __import__(
    "test_launch_smolvla_piper_schema6_autonomous_watcher"
)


def _code_string_constants(code: types.CodeType) -> list[str]:
    values: list[str] = []
    for value in code.co_consts:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, types.CodeType):
            values.extend(_code_string_constants(value))
    return values


def test_r13_deployment_constants_bind_measured_source_and_lobo() -> None:
    assert watcher.EXPECTED_LOBO_LAUNCHER_SHA256 == (
        "3af8933fa5ccd09e7b06dc1912926510e5a9fb0508b2aee3c9d323adafb71206"
    )
    assert watcher.EXPECTED_LOBO_STATIC_PLAN_SHA256 == (
        "e847ea6773cddcf0a675fcd77210d57ea267afb6a1b50d6d9dbc2158dca06dc4"
    )
    assert watcher.EXPECTED_SOURCE_ROOT == Path(
        "/home/user/etsf_smolvla_schema5_native_source_training_r13_20260829"
    )
    assert watcher.EXPECTED_SOURCE_PLAN_SHA256 == (
        "06f3a90db413cce334b1d34a4e09fbba7a3159abc7ac830f8d2bb981cbf404e5"
    )
    assert watcher.EXPECTED_SOURCE_STATIC_PLAN_SHA256 == (
        "47a1af14720bf86789276101159368764c4a10b8c92efc4aec9025bf947f306f"
    )
    assert watcher.EXPECTED_SOURCE_LAUNCHER_SHA256 == (
        "bc11b909d82bb86349f0557a94222b4100e939cfca95d08a5f6a5f49a38dbc90"
    )
    assert watcher.EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256 == (
        "491e92026c76430464c352aaae1f392df4e842ee8b24ccd382ed73e9fc8e540c"
    )
    assert watcher.DESIGNATED_LOBO_ROOT == Path(
        "/home/user/etsf_multibody_lobo_autonomous_r13_20260829"
    )
    assert watcher.DESIGNATED_LOBO_OUTPUTS == {
        "piper": Path(
            "/home/user/etsf_multibody_lobo_piper_train_r13_20260829"
        ),
        "ur5-wsg": Path(
            "/home/user/etsf_multibody_lobo_ur5_train_r13_20260829"
        ),
    }


def test_r13_binding_is_content_addressed_and_excludes_r12() -> None:
    binding = watcher.r13_deployment_binding()
    assert binding["format"] == watcher.R13_BINDING_FORMAT
    assert binding["base_implementation"]["sha256"] == (
        watcher.R13_BASE_IMPLEMENTATION_SHA256
    )
    assert watcher.canonical_sha256(binding) == (
        watcher.R13_DEPLOYMENT_BINDING_SHA256
    )
    assert "r12_20260828" not in json.dumps(binding, sort_keys=True)


def test_r13_static_preflight_embeds_binding_and_rehashes_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        watcher,
        "_r13_base_static_preflight",
        lambda _args: {
            "format": watcher.FORMAT,
            "status": "synthetic_base_preflight",
            "static_plan_sha256": "stale",
        },
    )
    plan = watcher.static_preflight(argparse.Namespace())
    unsigned = dict(plan)
    logical = unsigned.pop("static_plan_sha256")
    assert logical == watcher.canonical_sha256(unsigned)
    assert plan["r13_deployment_binding"] == watcher.r13_deployment_binding()
    assert plan["r13_deployment_binding_sha256"] == (
        watcher.R13_DEPLOYMENT_BINDING_SHA256
    )


def test_old_r12_lobo_root_is_rejected_before_any_wait(tmp_path: Path) -> None:
    with pytest.raises(
        watcher.WatcherContractError,
        match="differs from designated aggregate root",
    ):
        watcher.wait_for_lobo_completion(
            Path("/home/user/etsf_multibody_lobo_autonomous_r12_20260828"),
            state={},
            state_path=tmp_path / "state.json",
            poll_seconds=0.01,
            timeout_seconds=0.01,
            sleep=lambda _seconds: None,
        )
    assert not (tmp_path / "state.json").exists()


def test_r13_binding_accepts_complete_content_addressed_lobo_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(legacy_fixtures, "watcher", watcher)
    root = legacy_fixtures._lobo_fixture(tmp_path, monkeypatch)
    audit = watcher.validate_lobo_terminal_summary(root)
    assert audit["status"] == "verified_piper_then_ur5_lobo_terminal_exit_zero"
    assert audit["execution_order"] == [
        stage for stage, _body in watcher.LOBO_STAGES
    ]
    assert audit["lobo_checkpoints_rerank_authorized"] is False


def test_r13_lobo_gate_fails_closed_when_source_checkpoint_is_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(legacy_fixtures, "watcher", watcher)
    root = legacy_fixtures._lobo_fixture(tmp_path, monkeypatch)
    terminal = watcher.load_json(root / "final_receipt.json", role="test terminal")
    checkpoint = Path(terminal["source63_audit"]["ensemble_checkpoint"])
    original = checkpoint.read_bytes()
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(original + b"tampered")
    checkpoint.chmod(0o444)
    with pytest.raises(
        watcher.WatcherContractError,
        match="bound native source artifacts changed",
    ):
        watcher.validate_lobo_terminal_summary(root)


def test_r13_base_implementation_sha_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tampered = tmp_path / watcher.R13_BASE_IMPLEMENTATION_FILENAME
    tampered.write_text("# changed implementation\n", encoding="utf-8")
    monkeypatch.setattr(watcher, "_R13_BASE_IMPLEMENTATION_PATH", tampered)
    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        watcher._verify_r13_base_implementation()


def test_compiled_r13_binding_code_has_no_stale_r7h_wording() -> None:
    strings = _code_string_constants(
        watcher._validate_lobo_native_source_binding.__code__
    )
    assert not any("r7h" in value.lower() for value in strings)
