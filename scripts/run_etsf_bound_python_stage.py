#!/usr/bin/env python3
"""Run one ETSF stage under a frozen isolated Python/Torch runtime.

This wrapper is intentionally standard-library-only until it imports Torch.
It is launched with ``python -I``; therefore inherited PYTHONPATH, the current
directory, and the user site cannot select Torch.  Only after Torch and its
binary module have been authenticated against the immutable launch plan is the
single approved target script directory inserted into ``sys.path``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping


RUNTIME_CONTRACT_FORMAT = "etsf_isolated_python_torch_runtime_v1"
PLAN_FORMAT = "etsf_smolvla_schema5_source63_native_training_launcher_v1"
FORCED_PYTHON_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
}
SCRUBBED_ENVIRONMENT_NAMES = frozenset(
    {
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CONDA_PYTHON_EXE",
        "__PYVENV_LAUNCHER__",
    }
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("launch plan must be a materialized regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != PLAN_FORMAT:
        raise RuntimeError("launch plan is invalid")
    claimed = value.get("static_plan_sha256")
    unsigned = dict(value)
    unsigned.pop("static_plan_sha256", None)
    if claimed != canonical_sha256(unsigned) or claimed != expected_sha256:
        raise RuntimeError("launch plan SHA256 changed")
    return value


def _current_runtime() -> dict[str, Any]:
    # Import before adding the target script directory: the authenticated venv
    # site-packages are the only non-stdlib import source at this point.
    import torch

    def module_file(path_value: str) -> dict[str, str]:
        path = Path(path_value).resolve(strict=True)
        return {"path": str(path), "sha256": file_sha256(path)}

    value: dict[str, Any] = {
        "format": RUNTIME_CONTRACT_FORMAT,
        "isolated": bool(sys.flags.isolated),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": str(sys.version),
        "python_prefix": str(Path(sys.prefix).resolve(strict=True)),
        "python_base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "torch_module": module_file(str(torch.__file__)),
        "torch_c_module": module_file(str(torch._C.__file__)),
    }
    value["runtime_contract_sha256"] = canonical_sha256(value)
    return value


def _verify_environment(plan: Mapping[str, Any]) -> None:
    contract = plan.get("environment_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("launch plan lacks the environment contract")
    unexpected = sorted(
        key
        for key in os.environ
        if key.upper().startswith("PYTHON") and key not in FORCED_PYTHON_ENVIRONMENT
    )
    wrong = {
        key: os.environ.get(key)
        for key, expected in FORCED_PYTHON_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    leaked = sorted(key for key in SCRUBBED_ENVIRONMENT_NAMES if key in os.environ)
    if unexpected or wrong or leaked:
        raise RuntimeError(
            "stage inherited a non-canonical Python environment: "
            f"unexpected={unexpected}, wrong={wrong}, leaked={leaked}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != contract.get(
        "cuda_visible_devices"
    ) or os.environ.get("OMP_NUM_THREADS") != contract.get("omp_num_threads"):
        raise RuntimeError("stage CUDA/OMP environment differs from launch plan")


def run_bound_target(
    *, launch_plan: Path, static_plan_sha256: str, target: Path, target_argv: list[str]
) -> None:
    plan = _load_plan(launch_plan, static_plan_sha256)
    _verify_environment(plan)
    expected_runtime = plan.get("runtime_contract")
    if not isinstance(expected_runtime, Mapping):
        raise RuntimeError("launch plan lacks the Python/Torch runtime contract")
    actual_runtime = _current_runtime()
    if actual_runtime != dict(expected_runtime):
        raise RuntimeError(
            "bound stage Python/Torch runtime drifted: "
            f"expected torch={expected_runtime.get('torch_version')} at "
            f"{dict(expected_runtime.get('torch_module', {})).get('path')}, "
            f"found torch={actual_runtime.get('torch_version')} at "
            f"{actual_runtime['torch_module']['path']}"
        )
    approved = {
        Path(str(plan["initializer"])).resolve(strict=True),
        Path(str(plan["trainer"])).resolve(strict=True),
    }
    resolved_target = target.resolve(strict=True)
    if resolved_target not in approved or not resolved_target.is_file():
        raise RuntimeError("stage target is not approved by the launch plan")
    scripts = (Path(str(plan["code_root"])) / "scripts").resolve(strict=True)
    if resolved_target.parent != scripts:
        raise RuntimeError("stage target escaped the immutable scripts directory")
    if (scripts / "torch.py").exists() or (scripts / "torch").exists():
        raise RuntimeError("immutable scripts directory contains a Torch shadow")
    sys.path.insert(0, str(scripts))
    sys.argv = [str(resolved_target), *target_argv]
    runpy.run_path(str(resolved_target), run_name="__main__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--static-plan-sha256", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("target_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.target_argv[:1] == ["--"]:
        args.target_argv = args.target_argv[1:]
    return args


def main() -> None:
    args = parse_args()
    run_bound_target(
        launch_plan=args.launch_plan,
        static_plan_sha256=args.static_plan_sha256,
        target=args.target,
        target_argv=args.target_argv,
    )


if __name__ == "__main__":
    main()
