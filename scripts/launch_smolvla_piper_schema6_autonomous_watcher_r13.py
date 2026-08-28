#!/usr/bin/env python3
"""Content-addressed Source-r13/LOBO-r13 Piper schema6 watcher.

This is a versioned deployment entrypoint for the existing schema6 watcher,
not a mutable rewrite of the r12 authority.  It verifies the exact bytes of
the reviewed base implementation before executing them, replaces only stale
human-readable r7h role names, and then freezes the measured Source r13 and
LOBO r13 deployment bindings below.

The sibling base implementation is required at runtime.  Both files must be
deployed read-only in the same immutable code snapshot.
"""

from __future__ import annotations

import hashlib as _r13_hashlib
from pathlib import Path as _R13Path
from typing import Any as _R13Any, Mapping as _R13Mapping


R13_BASE_IMPLEMENTATION_FILENAME = (
    "launch_smolvla_piper_schema6_autonomous_watcher.py"
)
R13_BASE_IMPLEMENTATION_SHA256 = (
    "d6ea99f26a10dfae624d1bd7a588a3ad2960c94e3be557c5150879d0ab0dff48"
)
R13_LOBO_LAUNCHER_SHA256 = (
    "3af8933fa5ccd09e7b06dc1912926510e5a9fb0508b2aee3c9d323adafb71206"
)
R13_LOBO_STATIC_PLAN_SHA256 = (
    "e847ea6773cddcf0a675fcd77210d57ea267afb6a1b50d6d9dbc2158dca06dc4"
)
R13_SOURCE_ROOT = _R13Path(
    "/home/user/etsf_smolvla_schema5_native_source_training_r13_20260829"
)
R13_SOURCE_PLAN_SHA256 = (
    "06f3a90db413cce334b1d34a4e09fbba7a3159abc7ac830f8d2bb981cbf404e5"
)
R13_SOURCE_STATIC_PLAN_SHA256 = (
    "47a1af14720bf86789276101159368764c4a10b8c92efc4aec9025bf947f306f"
)
R13_SOURCE_LAUNCHER_SHA256 = (
    "bc11b909d82bb86349f0557a94222b4100e939cfca95d08a5f6a5f49a38dbc90"
)
R13_SOURCE_IMPLEMENTATION_BUNDLE_SHA256 = (
    "491e92026c76430464c352aaae1f392df4e842ee8b24ccd382ed73e9fc8e540c"
)
R13_LOBO_ROOT = _R13Path(
    "/home/user/etsf_multibody_lobo_autonomous_r13_20260829"
)
R13_LOBO_OUTPUTS = {
    "piper": _R13Path(
        "/home/user/etsf_multibody_lobo_piper_train_r13_20260829"
    ),
    "ur5-wsg": _R13Path(
        "/home/user/etsf_multibody_lobo_ur5_train_r13_20260829"
    ),
}
R13_BINDING_FORMAT = "etsf_smolvla_piper_schema6_r13_deployment_binding_v1"

_R13_ENTRYPOINT_PATH = _R13Path(__file__).resolve(strict=True)
_R13_BASE_IMPLEMENTATION_PATH = (
    _R13_ENTRYPOINT_PATH.parent / R13_BASE_IMPLEMENTATION_FILENAME
)


def _read_r13_base_implementation() -> bytes:
    path = _R13_BASE_IMPLEMENTATION_PATH
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "r13 schema6 base watcher must be a materialized regular file"
        )
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("r13 schema6 base watcher changed while it was read")
    actual = _r13_hashlib.sha256(raw).hexdigest()
    if actual != R13_BASE_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "r13 schema6 base watcher SHA-256 differs from its frozen binding"
        )
    return raw


def _verify_r13_base_implementation() -> None:
    _read_r13_base_implementation()


_r13_base_source = _read_r13_base_implementation().decode("utf-8")
for _r13_stale_wording, _r13_current_wording in {
    "designated r7h source root": "designated Source r13 root",
    "designated r7h": "designated Source r13",
    "bound r7h source launch plan": "bound Source r13 launch plan",
}.items():
    _r13_base_source = _r13_base_source.replace(
        _r13_stale_wording, _r13_current_wording
    )
if "r7h" in _r13_base_source.lower():
    raise RuntimeError("stale r7h wording remains in the r13 watcher implementation")

# Execute the reviewed implementation in this module's namespace.  Suppress
# its original __main__ block; the versioned entrypoint calls main only after
# all measured deployment constants and integrity wrappers are installed.
_r13_public_module_name = __name__
__name__ = "_etsf_smolvla_piper_schema6_autonomous_watcher_r13_impl"
try:
    exec(
        compile(
            _r13_base_source,
            str(_R13_BASE_IMPLEMENTATION_PATH),
            "exec",
        ),
        globals(),
    )
finally:
    __name__ = _r13_public_module_name


# Content-addressed deployment authority.  The base implementation resolves
# these names dynamically, so every validation and wait gate now binds r13.
EXPECTED_LOBO_LAUNCHER_SHA256 = R13_LOBO_LAUNCHER_SHA256
EXPECTED_LOBO_STATIC_PLAN_SHA256 = R13_LOBO_STATIC_PLAN_SHA256
EXPECTED_SOURCE_ROOT = R13_SOURCE_ROOT
EXPECTED_SOURCE_PLAN_SHA256 = R13_SOURCE_PLAN_SHA256
EXPECTED_SOURCE_STATIC_PLAN_SHA256 = R13_SOURCE_STATIC_PLAN_SHA256
EXPECTED_SOURCE_LAUNCHER_SHA256 = R13_SOURCE_LAUNCHER_SHA256
EXPECTED_SOURCE_IMPLEMENTATION_BUNDLE_SHA256 = (
    R13_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
)
DESIGNATED_LOBO_ROOT = R13_LOBO_ROOT
DESIGNATED_LOBO_OUTPUTS = dict(R13_LOBO_OUTPUTS)


def r13_deployment_binding() -> dict[str, _R13Any]:
    """Return a fresh JSON-safe projection of the frozen r13 authority."""

    return {
        "format": R13_BINDING_FORMAT,
        "base_implementation": {
            "relative_path": R13_BASE_IMPLEMENTATION_FILENAME,
            "sha256": R13_BASE_IMPLEMENTATION_SHA256,
        },
        "lobo": {
            "launcher_sha256": R13_LOBO_LAUNCHER_SHA256,
            "static_plan_sha256": R13_LOBO_STATIC_PLAN_SHA256,
            "root": str(R13_LOBO_ROOT),
            "outputs": {
                body: str(path) for body, path in sorted(R13_LOBO_OUTPUTS.items())
            },
        },
        "source": {
            "root": str(R13_SOURCE_ROOT),
            "launch_plan_file_sha256": R13_SOURCE_PLAN_SHA256,
            "static_plan_sha256": R13_SOURCE_STATIC_PLAN_SHA256,
            "launcher_sha256": R13_SOURCE_LAUNCHER_SHA256,
            "implementation_bundle_sha256": (
                R13_SOURCE_IMPLEMENTATION_BUNDLE_SHA256
            ),
        },
        "lobo_checkpoints_rerank_authorized": False,
        "deployment_rerank_authority": "native_source_ensemble_only",
    }


R13_DEPLOYMENT_BINDING_SHA256 = canonical_sha256(r13_deployment_binding())

_r13_base_static_preflight = static_preflight
_r13_base_verify_static_bindings = verify_static_bindings


def _validate_r13_plan_binding(plan: _R13Mapping[str, _R13Any]) -> None:
    binding = plan.get("r13_deployment_binding")
    binding_sha256 = plan.get("r13_deployment_binding_sha256")
    expected = r13_deployment_binding()
    if (
        binding != expected
        or binding_sha256 != R13_DEPLOYMENT_BINDING_SHA256
        or binding_sha256 != canonical_sha256(expected)
    ):
        raise WatcherContractError(
            "schema6 static plan lost its Source-r13/LOBO-r13 deployment binding"
        )


def static_preflight(args: argparse.Namespace) -> dict[str, _R13Any]:
    """Run the inherited preflight and add the explicit r13 authority hash."""

    _verify_r13_base_implementation()
    plan = dict(_r13_base_static_preflight(args))
    plan.pop("static_plan_sha256", None)
    plan["r13_deployment_binding"] = r13_deployment_binding()
    plan["r13_deployment_binding_sha256"] = R13_DEPLOYMENT_BINDING_SHA256
    plan["static_plan_sha256"] = canonical_sha256(plan)
    _validate_r13_plan_binding(plan)
    return plan


def verify_static_bindings(plan: _R13Mapping[str, _R13Any]) -> None:
    """Revalidate both the inherited closure and the versioned base/binding."""

    _verify_r13_base_implementation()
    _validate_r13_plan_binding(plan)
    _r13_base_verify_static_bindings(plan)


__all__ = list(globals().get("__all__", [])) + [
    "R13_BASE_IMPLEMENTATION_FILENAME",
    "R13_BASE_IMPLEMENTATION_SHA256",
    "R13_BINDING_FORMAT",
    "R13_DEPLOYMENT_BINDING_SHA256",
    "R13_LOBO_LAUNCHER_SHA256",
    "R13_LOBO_OUTPUTS",
    "R13_LOBO_ROOT",
    "R13_LOBO_STATIC_PLAN_SHA256",
    "R13_SOURCE_IMPLEMENTATION_BUNDLE_SHA256",
    "R13_SOURCE_LAUNCHER_SHA256",
    "R13_SOURCE_PLAN_SHA256",
    "R13_SOURCE_ROOT",
    "R13_SOURCE_STATIC_PLAN_SHA256",
    "r13_deployment_binding",
]


if _r13_public_module_name == "__main__":
    main()
