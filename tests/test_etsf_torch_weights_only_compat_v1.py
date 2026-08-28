from __future__ import annotations

import _codecs
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from etsf_torch_weights_only_compat_v1 import (  # noqa: E402
    FORMAT,
    WeightsOnlyCompatibilityError,
    load_numpy_weights_only,
    load_weights_only,
    numpy_weights_only_safe_globals,
)


class _LegacySerialization:
    def __init__(self, initial=()):
        self.values = list(initial)

    def get_safe_globals(self):
        return list(self.values)

    def add_safe_globals(self, values):
        for value in values:
            if not any(value is existing for existing in self.values):
                self.values.append(value)

    def clear_safe_globals(self):
        self.values.clear()


class _SafeGlobalsContext:
    def __init__(self, serialization, requested):
        self.serialization = serialization
        self.requested = requested
        self.before = None

    def __enter__(self):
        self.before = list(self.serialization.values)
        self.serialization.add_safe_globals(self.requested)
        return self

    def __exit__(self, *_exc):
        self.serialization.values[:] = self.before


class _ModernSerialization(_LegacySerialization):
    def safe_globals(self, requested):
        return _SafeGlobalsContext(self, requested)


def test_contract_is_explicitly_versioned() -> None:
    assert FORMAT == "etsf_torch_weights_only_compat_v1"


def test_numpy_allowlist_includes_legacy_codecs_encoder() -> None:
    assert any(
        value is _codecs.encode for value in numpy_weights_only_safe_globals()
    )


@pytest.mark.parametrize("modern", [False, True])
def test_old_and_new_safe_global_apis_load_weights_only_and_restore(modern) -> None:
    prior = object()
    allowed = object()
    serialization = (
        _ModernSerialization([prior]) if modern else _LegacySerialization([prior])
    )
    calls = []

    def load(path, *, map_location, weights_only):
        calls.append((path, map_location, weights_only))
        assert any(value is allowed for value in serialization.values)
        return {"verified": True}

    torch_module = SimpleNamespace(load=load, serialization=serialization)
    result = load_weights_only(
        "checkpoint.pt", allowed_globals=[allowed], torch_module=torch_module
    )

    assert result == {"verified": True}
    assert calls == [("checkpoint.pt", "cpu", True)]
    assert len(serialization.values) == 1
    assert serialization.values[0] is prior


def test_partial_safe_global_api_fails_before_load() -> None:
    calls = []

    def load(path, *, map_location, weights_only):
        calls.append(path)

    serialization = SimpleNamespace(
        add_safe_globals=lambda _values: None,
        get_safe_globals=lambda: [],
    )
    torch_module = SimpleNamespace(load=load, serialization=serialization)

    with pytest.raises(
        WeightsOnlyCompatibilityError,
        match="neither a complete modern nor legacy safe-global API",
    ):
        load_weights_only(
            "checkpoint.pt", allowed_globals=[object()], torch_module=torch_module
        )
    assert calls == []


def test_missing_weights_only_parameter_fails_closed_and_restores() -> None:
    prior = object()
    serialization = _LegacySerialization([prior])
    calls = []

    def load(path, *, map_location):
        calls.append(path)

    torch_module = SimpleNamespace(load=load, serialization=serialization)
    with pytest.raises(WeightsOnlyCompatibilityError, match="weights_only parameter"):
        load_weights_only(
            "checkpoint.pt", allowed_globals=[object()], torch_module=torch_module
        )

    assert calls == []
    assert len(serialization.values) == 1
    assert serialization.values[0] is prior


def test_load_failure_restores_legacy_process_global_allowlist() -> None:
    prior = object()
    serialization = _LegacySerialization([prior])

    def load(path, *, map_location, weights_only):
        raise ValueError("malformed weights-only checkpoint")

    torch_module = SimpleNamespace(load=load, serialization=serialization)
    with pytest.raises(ValueError, match="malformed weights-only checkpoint"):
        load_weights_only(
            "checkpoint.pt", allowed_globals=[object()], torch_module=torch_module
        )

    assert len(serialization.values) == 1
    assert serialization.values[0] is prior


def test_safe_global_drift_during_load_fails_closed_and_restores() -> None:
    prior = object()
    requested = object()
    drift = object()
    serialization = _LegacySerialization([prior])

    def load(path, *, map_location, weights_only):
        serialization.values.append(drift)
        return {"must_not_be_returned": True}

    torch_module = SimpleNamespace(load=load, serialization=serialization)
    with pytest.raises(WeightsOnlyCompatibilityError, match="changed during"):
        load_weights_only(
            "checkpoint.pt",
            allowed_globals=[requested],
            torch_module=torch_module,
        )

    assert len(serialization.values) == 1
    assert serialization.values[0] is prior


def test_real_numpy_checkpoint_round_trip_is_weights_only(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    checkpoint = tmp_path / "numpy_checkpoint.pt"
    expected = np.asarray([1.5, 2.5], dtype=np.float32)
    torch.save({"array": expected}, checkpoint)

    loaded = load_numpy_weights_only(checkpoint)

    np.testing.assert_array_equal(loaded["array"], expected)


def test_source_launcher_uses_versioned_loader_without_unsafe_fallback() -> None:
    source = (
        SCRIPTS / "launch_smolvla_schema5_source63_native_training.py"
    ).read_text(encoding="utf-8")
    assert '"etsf_torch_weights_only_compat_v1.py"' in source
    assert "load_numpy_weights_only(member_path)" in source
    assert "load_numpy_weights_only(ensemble_path)" in source
    assert "torch.serialization.safe_globals" not in source
    assert "weights_only=False" not in source
