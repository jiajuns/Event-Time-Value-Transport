#!/usr/bin/env python3
"""Versioned, fail-closed compatibility for PyTorch weights-only loading.

PyTorch 2.4 exposes ``add/get/clear_safe_globals`` but not the later
``safe_globals`` context manager.  This module supports both APIs while never
falling back to unrestricted pickle loading.  The legacy API is treated as a
process-global critical section: the previous allowlist is snapshotted,
checked for unexpected mutation, and restored before a loaded value is
returned.
"""

from __future__ import annotations

import importlib
import inspect
import threading
import _codecs
from pathlib import Path
from typing import Any, Iterable, Sequence


FORMAT = "etsf_torch_weights_only_compat_v1"
_SAFE_GLOBALS_LOCK = threading.RLock()


class WeightsOnlyCompatibilityError(RuntimeError):
    """Raised when a verified weights-only load cannot be guaranteed."""


def _identity_collection_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare process-global allowlists without relying on API list order."""

    return (
        len(left) == len(right)
        and all(_contains_identity(right, value) for value in left)
        and all(_contains_identity(left, value) for value in right)
    )


def _contains_identity(values: Sequence[Any], expected: Any) -> bool:
    return any(value is expected for value in values)


def _validated_allowed_globals(values: Iterable[Any]) -> tuple[Any, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise WeightsOnlyCompatibilityError(
            "weights-only safe globals must be a finite iterable"
        ) from error
    for index, value in enumerate(result):
        if any(value is previous for previous in result[:index]):
            raise WeightsOnlyCompatibilityError(
                "weights-only safe globals contain duplicate identities"
            )
    return result


def _snapshot(getter: Any) -> tuple[Any, ...]:
    if not callable(getter):
        raise WeightsOnlyCompatibilityError(
            "torch safe-global snapshot API is unavailable"
        )
    try:
        return tuple(getter())
    except BaseException as error:
        raise WeightsOnlyCompatibilityError(
            "torch safe-global snapshot failed closed"
        ) from error


def _assert_requested_installed(
    installed: Sequence[Any], requested: Sequence[Any]
) -> None:
    if not all(_contains_identity(installed, value) for value in requested):
        raise WeightsOnlyCompatibilityError(
            "torch did not install every requested weights-only safe global"
        )


def _strict_torch_load(torch_module: Any, checkpoint: Any) -> Any:
    loader = getattr(torch_module, "load", None)
    if not callable(loader):
        raise WeightsOnlyCompatibilityError("torch.load is unavailable")
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError) as error:
        raise WeightsOnlyCompatibilityError(
            "torch.load signature cannot prove weights_only support"
        ) from error
    if "weights_only" not in signature.parameters:
        raise WeightsOnlyCompatibilityError(
            "torch.load does not expose the required weights_only parameter"
        )
    return loader(checkpoint, map_location="cpu", weights_only=True)


def _load_with_context_api(
    torch_module: Any,
    requested: tuple[Any, ...],
    context_factory: Any,
    getter: Any,
    checkpoint: Any,
) -> Any:
    before = _snapshot(getter)
    value: Any = None
    operation_error: BaseException | None = None
    try:
        with context_factory(list(requested)):
            installed = _snapshot(getter)
            _assert_requested_installed(installed, requested)
            value = _strict_torch_load(torch_module, checkpoint)
            after_load = _snapshot(getter)
            if not _identity_collection_equal(after_load, installed):
                raise WeightsOnlyCompatibilityError(
                    "torch safe globals changed during weights-only load"
                )
    except BaseException as error:
        operation_error = error

    try:
        restored = _snapshot(getter)
        if not _identity_collection_equal(restored, before):
            raise WeightsOnlyCompatibilityError(
                "torch safe_globals context did not restore its prior state"
            )
    except BaseException as restore_error:
        if operation_error is not None:
            raise WeightsOnlyCompatibilityError(
                "weights-only load and safe-global restoration both failed closed"
            ) from restore_error
        raise
    if operation_error is not None:
        raise operation_error
    return value


def _load_with_legacy_api(
    torch_module: Any,
    requested: tuple[Any, ...],
    adder: Any,
    getter: Any,
    clearer: Any,
    checkpoint: Any,
) -> Any:
    before = _snapshot(getter)
    value: Any = None
    operation_error: BaseException | None = None
    try:
        adder(list(requested))
        installed = _snapshot(getter)
        _assert_requested_installed(installed, requested)
        value = _strict_torch_load(torch_module, checkpoint)
        after_load = _snapshot(getter)
        if not _identity_collection_equal(after_load, installed):
            raise WeightsOnlyCompatibilityError(
                "legacy torch safe globals changed during weights-only load"
            )
    except BaseException as error:
        operation_error = error

    restore_error: BaseException | None = None
    try:
        clearer()
        if before:
            adder(list(before))
        restored = _snapshot(getter)
        if not _identity_collection_equal(restored, before):
            raise WeightsOnlyCompatibilityError(
                "legacy torch safe globals were not restored exactly"
            )
    except BaseException as error:
        restore_error = error

    if restore_error is not None:
        raise WeightsOnlyCompatibilityError(
            "legacy torch safe-global restoration failed closed"
        ) from restore_error
    if operation_error is not None:
        raise operation_error
    return value


def load_weights_only(
    checkpoint: str | Path | Any,
    *,
    allowed_globals: Iterable[Any] = (),
    torch_module: Any | None = None,
) -> Any:
    """Load on CPU with an explicit ``weights_only=True`` security contract.

    A non-empty allowlist requires either the modern context API or the full
    legacy add/get/clear API.  Missing or partial APIs abort before loading.
    """

    requested = _validated_allowed_globals(allowed_globals)
    torch_value = torch_module or importlib.import_module("torch")
    if not requested:
        return _strict_torch_load(torch_value, checkpoint)

    # Both APIs mutate a process-global allowlist.  Serialize the complete
    # snapshot/install/load/restore transaction so two validation threads can
    # never observe or clear each other's temporary globals.
    with _SAFE_GLOBALS_LOCK:
        serialization = getattr(torch_value, "serialization", None)
        if serialization is None:
            raise WeightsOnlyCompatibilityError(
                "torch.serialization is unavailable for an allowlisted load"
            )
        context_factory = getattr(serialization, "safe_globals", None)
        getter = getattr(serialization, "get_safe_globals", None)
        if callable(context_factory) and callable(getter):
            return _load_with_context_api(
                torch_value, requested, context_factory, getter, checkpoint
            )

        adder = getattr(serialization, "add_safe_globals", None)
        clearer = getattr(serialization, "clear_safe_globals", None)
        if callable(adder) and callable(getter) and callable(clearer):
            return _load_with_legacy_api(
                torch_value, requested, adder, getter, clearer, checkpoint
            )
        raise WeightsOnlyCompatibilityError(
            "torch has neither a complete modern nor legacy safe-global API"
        )


def numpy_weights_only_safe_globals(numpy_module: Any | None = None) -> tuple[Any, ...]:
    """Return the minimal NumPy globals used by ETSF checkpoint metadata."""

    np = numpy_module or importlib.import_module("numpy")
    values = (
        _codecs.encode,
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
    )
    return _validated_allowed_globals(values)


def load_numpy_weights_only(
    checkpoint: str | Path | Any,
    *,
    torch_module: Any | None = None,
    numpy_module: Any | None = None,
) -> Any:
    """Load an ETSF NumPy-bearing checkpoint under the versioned contract."""

    return load_weights_only(
        checkpoint,
        allowed_globals=numpy_weights_only_safe_globals(numpy_module),
        torch_module=torch_module,
    )
