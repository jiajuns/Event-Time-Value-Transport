#!/usr/bin/env python3
"""Environment-audited Source-r13/LOBO-r13 Piper schema6 watcher r14.

R14 preserves the reviewed r13 fail-closed pipeline while closing one missing
preflight gate: the selected Python environment must successfully import every
direct third-party dependency needed by the schema6 implementation before an
output root, GPU lock, simulator, or HDF5 payload can be touched.

The interpreter is configurable through the inherited ``--python-bin`` flag,
but its invocation path, resolved executable bytes, isolated ``sys.path``,
package versions, module bytes, distribution metadata/RECORD commitments, and
security-relevant environment projection are content-addressed in the static
plan and revalidated before every stage.  The failed/frozen r13 schema6 output
is never resumed or reused; r14 is pinned to its own absent output root.
"""

from __future__ import annotations

import hashlib as _r14_hashlib
import json as _r14_json
import os as _r14_os
import subprocess as _r14_subprocess
import sys as _r14_sys
from pathlib import Path as _R14Path
from typing import Any as _R14Any, Mapping as _R14Mapping, Sequence as _R14Sequence


R14_PARENT_IMPLEMENTATION_FILENAME = (
    "launch_smolvla_piper_schema6_autonomous_watcher_r13.py"
)
R14_PARENT_IMPLEMENTATION_SHA256 = (
    "dc548a5a8155dfd479da521f41c033417c3bfb260011f2f54865282fd1952da1"
)
R14_BINDING_FORMAT = "etsf_smolvla_piper_schema6_r14_environment_binding_v2"
R14_PYTHON_ENVIRONMENT_FORMAT = (
    "etsf_smolvla_piper_schema6_python_environment_audit_v2"
)
R14_PYTHON_PROBE_FORMAT = "etsf_schema6_python_dependency_probe_v2"
R14_IMPORT_CONTRACT_FORMAT = "etsf_schema6_import_only_closure_v1"
R14_REQUIRED_PYTHON_IMPORTS = (
    "numpy",
    "torch",
    "omegaconf",
    "antlr4",
    "h5py",
    "cv2",
    "yaml",
)
R14_REQUIRED_PYTHON_VERSIONS = {
    "omegaconf": "2.3.0",
    "antlr4": "4.9.3",
}
R14_LOCAL_IMPORT_TARGETS = (
    {
        "import_name": "materialize_smolvla_piper_schema6_reset_contract",
        "relative_path": (
            "scripts/materialize_smolvla_piper_schema6_reset_contract.py"
        ),
    },
    {
        "import_name": "launch_smolvla_piper_schema6_development_collection",
        "relative_path": (
            "scripts/launch_smolvla_piper_schema6_development_collection.py"
        ),
    },
    {
        "import_name": "collect_openvla_etsf_rollouts",
        "relative_path": "scripts/collect_openvla_etsf_rollouts.py",
    },
)
R14_RUNTIME_IMPORT_TARGETS = (
    {
        "import_name": "rlinf.envs.robotwin.robotwin_env",
        "artifact_role": "rlinf_robotwin_env",
        "symbol": "RoboTwinEnv",
        "artifact_registry": "r6d",
    },
    {
        "import_name": "robotwin.envs.vector_env",
        "artifact_role": "robotwin_vector_env",
        "symbol": None,
        "artifact_registry": "r6d",
    },
    {
        "import_name": "envs._base_task",
        "artifact_role": "robotwin_base_task",
        "symbol": None,
        "artifact_registry": "r6d",
    },
    {
        "import_name": "envs.robot.robot",
        "artifact_role": "robotwin_robot_controller",
        "symbol": None,
        "artifact_registry": "r6d",
    },
    {
        "import_name": "lerobot.policies.factory",
        "artifact_role": "policy_factory",
        "symbol": "make_pre_post_processors",
        "artifact_registry": "r6e",
    },
    {
        "import_name": "lerobot.policies.smolvla.modeling_smolvla",
        "artifact_role": "smolvla_modeling",
        "symbol": "SmolVLAPolicy",
        "artifact_registry": "r6e",
    },
    {
        "import_name": "lerobot.policies.smolvla.smolvlm_with_expert",
        "artifact_role": "smolvlm_bridge",
        "symbol": None,
        "artifact_registry": "r6e",
    },
)
R14_AUDITED_ENVIRONMENT_NAMES = (
    "ASSETS_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "ROBOT_PLATFORM",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
)
R14_CODE_ROOT = _R14Path(
    "/home/user/etsf_smolvla_piper_schema6_code_r14_20260829"
)
R14_FORBIDDEN_PRIOR_CODE_ROOTS = (
    _R14Path("/home/user/etsf_smolvla_piper_schema6_code_r6j_20260828"),
)
R14_SCHEMA6_OUTPUT_ROOT = _R14Path(
    "/home/user/etsf_smolvla_piper_schema6_autonomous_r14_20260829"
)
R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS = (
    _R14Path(
        "/home/user/etsf_smolvla_piper_schema6_autonomous_r13_20260829"
    ),
)
R14_DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS = 120.0
R14_REQUIRED_IMPLEMENTATION_SHA256 = {
    "scripts/collect_openvla_etsf_rollouts.py": (
        "b8a20dcf15dea31d7708cf90c208b260d410eabd681499ab6101e6cb3cf8d491"
    ),
}

_R14_ENTRYPOINT_PATH = _R14Path(__file__).resolve(strict=True)
_R14_PARENT_IMPLEMENTATION_PATH = (
    _R14_ENTRYPOINT_PATH.parent / R14_PARENT_IMPLEMENTATION_FILENAME
)


def _read_r14_parent_implementation() -> bytes:
    path = _R14_PARENT_IMPLEMENTATION_PATH
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            "r14 schema6 parent watcher must be a materialized regular file"
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
        raise RuntimeError("r14 schema6 parent watcher changed while it was read")
    if _r14_hashlib.sha256(raw).hexdigest() != R14_PARENT_IMPLEMENTATION_SHA256:
        raise RuntimeError(
            "r14 schema6 parent watcher SHA-256 differs from its frozen binding"
        )
    return raw


def _verify_r14_parent_implementation() -> None:
    _read_r14_parent_implementation()


_r14_parent_source = _read_r14_parent_implementation().decode("utf-8")
_r14_public_module_name = __name__
__name__ = "_etsf_smolvla_piper_schema6_autonomous_watcher_r14_impl"
try:
    exec(
        compile(
            _r14_parent_source,
            str(_R14_PARENT_IMPLEMENTATION_PATH),
            "exec",
        ),
        globals(),
    )
finally:
    __name__ = _r14_public_module_name


_R14_PYTHON_PROBE_SOURCE = r'''
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

FORMAT = "etsf_schema6_python_dependency_probe_v2"
HDF_SUFFIXES = (".hdf5", ".h5", ".hdf")
required = tuple(json.loads(sys.argv[1]))
environment_names = tuple(json.loads(sys.argv[2]))
import_contract = json.loads(sys.argv[3])
hdf5_open_attempts = 0


def audit(event, args):
    global hdf5_open_attempts
    if event != "open" or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    text = os.fsdecode(raw)
    if Path(text).suffix.casefold() in HDF_SUFFIXES:
        hdf5_open_attempts += 1
        raise RuntimeError("dependency probe attempted to open an HDF5 payload")


sys.addaudithook(audit)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(raw_path):
    supplied = Path(raw_path)
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError("dependency origin is not a regular file")
    raw = resolved.read_bytes()
    return {
        "invocation_path": str(supplied),
        "resolved_path": str(resolved),
        "sha256": sha256_bytes(raw),
        "size": len(raw),
    }


def pyvenv_config_record():
    path = Path(sys.prefix) / "pyvenv.cfg"
    if not path.exists():
        return {
            "present": False,
            "include_system_site_packages": False,
        }
    record = file_record(path)
    content = Path(record["resolved_path"]).read_text(encoding="utf-8")
    fields = {}
    for line in content.splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip().casefold().replace("-", "_")] = value.strip()
    raw_include = fields.get("include_system_site_packages")
    if raw_include is None or raw_include.casefold() not in {"true", "false"}:
        raise RuntimeError("pyvenv.cfg lacks canonical include-system-site-packages")
    return {
        **record,
        "present": True,
        "content_utf8": content,
        "parsed_fields": fields,
        "include_system_site_packages": raw_include.casefold() == "true",
    }


packages_to_distributions = importlib.metadata.packages_distributions()
dependencies = []
for import_name in required:
    module = importlib.import_module(import_name)
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError(
            f"required dependency has no materialized origin: {import_name}"
        )
    distributions = []
    for distribution_name in sorted(packages_to_distributions.get(import_name, ())):
        distribution = importlib.metadata.distribution(distribution_name)
        metadata_text = distribution.read_text("METADATA")
        record_text = distribution.read_text("RECORD")
        distributions.append({
            "name": distribution.metadata.get("Name", distribution_name),
            "version": distribution.version,
            "metadata_sha256": (
                sha256_bytes(metadata_text.encode("utf-8"))
                if metadata_text is not None else None
            ),
            "record_sha256": (
                sha256_bytes(record_text.encode("utf-8"))
                if record_text is not None else None
            ),
        })
    module_version = getattr(module, "__version__", None)
    if module_version is None and distributions:
        module_version = distributions[0]["version"]
    if module_version is None:
        raise RuntimeError(
            f"required dependency has no auditable version: {import_name}"
        )
    dependencies.append({
        "import_name": import_name,
        "module_version": str(module_version),
        "origin": file_record(origin),
        "distributions": distributions,
    })

for python_path in reversed(import_contract["python_paths"]):
    if python_path in sys.path:
        sys.path.remove(python_path)
    sys.path.insert(0, python_path)

local_imports = []
for target in import_contract["local_imports"]:
    module = importlib.import_module(target["import_name"])
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError("local import target has no materialized origin")
    local_imports.append({
        "import_name": target["import_name"],
        "relative_path": target["relative_path"],
        "origin": file_record(origin),
    })

runtime_imports = []
for target in import_contract["runtime_imports"]:
    module = importlib.import_module(target["import_name"])
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str) or not origin:
        raise RuntimeError("runtime import target has no materialized origin")
    symbol = target["symbol"]
    if symbol is not None and not callable(getattr(module, symbol, None)):
        raise RuntimeError(
            f"runtime import target lacks callable symbol: {target['import_name']}"
        )
    runtime_imports.append({
        "import_name": target["import_name"],
        "artifact_role": target["artifact_role"],
        "symbol": symbol,
        "symbol_present": symbol is None or callable(getattr(module, symbol, None)),
        "origin": file_record(origin),
    })

executable = file_record(sys.executable)
payload = {
    "format": FORMAT,
    "status": "full_import_only_closure_verified_without_hdf5_access",
    "interpreter": {
        **executable,
        "version": sys.version,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "pyvenv_config": pyvenv_config_record(),
        "sys_path": list(sys.path),
    },
    "required_imports": list(required),
    "dependencies": dependencies,
    "import_contract_sha256": import_contract["contract_sha256"],
    "local_imports": local_imports,
    "runtime_imports": runtime_imports,
    "simulator_or_policy_objects_instantiated": 0,
    "environment_reset_calls": 0,
    "environment_step_calls": 0,
    "environment_projection": {
        name: os.environ.get(name) for name in environment_names
    },
    "hdf5_open_attempts": hdf5_open_attempts,
    "hdf5_payloads_opened": 0,
    "test_or_label_payloads_opened": 0,
    "output_roots_created": 0,
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
'''

R14_PYTHON_PROBE_SOURCE_SHA256 = _r14_hashlib.sha256(
    _R14_PYTHON_PROBE_SOURCE.encode("utf-8")
).hexdigest()

_r14_parent_static_preflight = static_preflight
_r14_parent_verify_static_bindings = verify_static_bindings
_r14_parent_add_common_arguments = add_common_arguments
_r14_parent_parse_args = parse_args
_r14_parent_common_run_argv = _common_run_argv


def r14_static_binding() -> dict[str, _R14Any]:
    return {
        "format": R14_BINDING_FORMAT,
        "parent_implementation": {
            "relative_path": R14_PARENT_IMPLEMENTATION_FILENAME,
            "sha256": R14_PARENT_IMPLEMENTATION_SHA256,
        },
        "code_root": str(R14_CODE_ROOT),
        "forbidden_prior_code_roots": [
            str(path) for path in R14_FORBIDDEN_PRIOR_CODE_ROOTS
        ],
        "schema6_output_root": str(R14_SCHEMA6_OUTPUT_ROOT),
        "forbidden_prior_schema6_output_roots": [
            str(path) for path in R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS
        ],
        "prior_schema6_output_reused": False,
        "required_python_imports": list(R14_REQUIRED_PYTHON_IMPORTS),
        "required_python_versions": dict(R14_REQUIRED_PYTHON_VERSIONS),
        "local_import_targets": [dict(value) for value in R14_LOCAL_IMPORT_TARGETS],
        "runtime_import_targets": [
            dict(value) for value in R14_RUNTIME_IMPORT_TARGETS
        ],
        "signed_runtime_imports_deferred": False,
        "audited_environment_names": list(R14_AUDITED_ENVIRONMENT_NAMES),
        "python_probe_source_sha256": R14_PYTHON_PROBE_SOURCE_SHA256,
        "required_implementation_sha256": dict(
            R14_REQUIRED_IMPLEMENTATION_SHA256
        ),
        "python_dependency_probe_before_output_creation": True,
        "python_dependency_probe_before_gpu_lock": True,
        "simulator_or_policy_objects_instantiated_by_python_probe": 0,
        "environment_reset_or_step_calls_by_python_probe": 0,
        "hdf5_payloads_opened_by_python_probe": 0,
        "test_or_label_payloads_opened_by_python_probe": 0,
    }


R14_STATIC_BINDING_SHA256 = canonical_sha256(r14_static_binding())


def _effective_environment_projection(
    environment: _R14Mapping[str, str],
) -> dict[str, str | None]:
    return {
        name: environment.get(name) for name in R14_AUDITED_ENVIRONMENT_NAMES
    }


def _validate_r14_code_root(code_root: _R14Path) -> None:
    supplied = reject_path_text(code_root, "r14 schema6 code root")
    if supplied in R14_FORBIDDEN_PRIOR_CODE_ROOTS:
        raise WatcherContractError("r14 cannot reuse the frozen r6j code root")
    if supplied != R14_CODE_ROOT:
        raise WatcherContractError(
            "r14 schema6 code root differs from its designated new root"
        )


def _bound_source_record(
    raw: _R14Any,
    *,
    role: str,
) -> dict[str, _R14Any]:
    if not isinstance(raw, _R14Mapping):
        raise WatcherContractError(f"signed runtime source is missing: {role}")
    path = resolve_existing(
        _R14Path(str(raw.get("path", ""))),
        role=f"signed runtime source {role}",
        directory=False,
    )
    expected_sha256 = raw.get("sha256")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not _is_sha256(expected_sha256)
        or _r14_hashlib.sha256(content).hexdigest() != expected_sha256
    ):
        raise WatcherContractError(f"signed runtime source changed: {role}")
    return {
        "path": str(path),
        "sha256": str(expected_sha256),
        "size": len(content),
        "read_consistency": "same_device_inode_size_mtime_before_and_after_read",
    }


def _build_r14_import_contract(
    plan: _R14Mapping[str, _R14Any],
) -> dict[str, _R14Any]:
    code_root = resolve_existing(
        _R14Path(str(plan.get("code_root", ""))),
        role="r14 schema6 immutable code root",
        directory=True,
    )
    _validate_r14_code_root(code_root)
    _require_read_only(code_root, "r14 schema6 immutable code root")
    scripts_root = resolve_existing(
        code_root / "scripts", role="r14 schema6 scripts", directory=True
    )
    _require_read_only(scripts_root, "r14 schema6 scripts")
    implementations = plan.get("implementation_files")
    if not isinstance(implementations, _R14Mapping):
        raise WatcherContractError("r14 implementation closure is missing")
    local_imports: list[dict[str, _R14Any]] = []
    for target in R14_LOCAL_IMPORT_TARGETS:
        relative_path = str(target["relative_path"])
        record = implementations.get(relative_path)
        if not isinstance(record, _R14Mapping):
            raise WatcherContractError(
                f"r14 local import escaped implementation closure: {relative_path}"
            )
        path = resolve_existing(
            _R14Path(str(record.get("path", ""))),
            role=f"r14 local import {relative_path}",
            directory=False,
        )
        expected_path = resolve_existing(
            code_root / relative_path,
            role=f"r14 local import source {relative_path}",
            directory=False,
        )
        if (
            path != expected_path
            or record.get("sha256") != file_sha256(path)
            or record.get("size") != path.stat().st_size
        ):
            raise WatcherContractError(
                f"r14 local import binding changed: {relative_path}"
            )
        local_imports.append(
            {
                **dict(target),
                "path": str(path),
                "sha256": str(record["sha256"]),
                "size": int(record["size"]),
            }
        )

    r6f_document, _projection = _load_signed_r6f_lineage_metadata(
        _R14Path(str(plan["r6f_preregistration"]))
    )
    inherited = r6f_document.get("inherited_R6e_contract")
    if not isinstance(inherited, _R14Mapping):
        raise WatcherContractError("r14 R6f runtime contract is missing")
    raw_roots = inherited.get("runtime_roots")
    r6e_sources = inherited.get("runtime_source_artifacts")
    r6d_binding = inherited.get("r6d_binding")
    r6d_sources = (
        r6d_binding.get("runtime_source_artifacts")
        if isinstance(r6d_binding, _R14Mapping)
        else None
    )
    if (
        not isinstance(raw_roots, _R14Mapping)
        or not isinstance(r6e_sources, _R14Mapping)
        or not isinstance(r6d_sources, _R14Mapping)
    ):
        raise WatcherContractError("r14 signed runtime import registries are missing")
    roots: dict[str, str] = {}
    for name in ("robotwin_root", "robotwin_code", "rlinf_root", "lerobot_root"):
        root = resolve_existing(
            _R14Path(str(raw_roots.get(name, ""))),
            role=f"r14 signed runtime root {name}",
            directory=True,
        )
        roots[name] = str(root)
    python_paths = [
        str(scripts_root),
        roots["robotwin_code"],
        roots["rlinf_root"],
        str(
            resolve_existing(
                _R14Path(roots["lerobot_root"]) / "src",
                role="r14 signed LeRobot source root",
                directory=True,
            )
        ),
    ]
    runtime_imports: list[dict[str, _R14Any]] = []
    for target in R14_RUNTIME_IMPORT_TARGETS:
        registry = r6d_sources if target["artifact_registry"] == "r6d" else r6e_sources
        source = _bound_source_record(
            registry.get(str(target["artifact_role"])),
            role=str(target["artifact_role"]),
        )
        runtime_imports.append({**dict(target), **source})
    contract: dict[str, _R14Any] = {
        "format": R14_IMPORT_CONTRACT_FORMAT,
        "code_root": str(code_root),
        "r6f_preregistration_sha256": plan["r6f_preregistration_sha256"],
        "runtime_roots": roots,
        "python_paths": python_paths,
        "local_imports": local_imports,
        "runtime_imports": runtime_imports,
        "import_only": True,
        "simulator_or_policy_objects_instantiated": 0,
        "environment_reset_calls": 0,
        "environment_step_calls": 0,
        "hdf5_payloads_opened": 0,
        "test_or_label_payloads_opened": 0,
    }
    _audit_embedded_paths(contract, "r14 import-only closure")
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _validate_probe_payload(
    payload: _R14Mapping[str, _R14Any],
    *,
    python: _R14Mapping[str, _R14Any],
    environment_projection: _R14Mapping[str, str | None],
    import_contract: _R14Mapping[str, _R14Any],
) -> dict[str, _R14Any]:
    _audit_embedded_paths(payload, "r14 Python dependency probe")
    interpreter = payload.get("interpreter")
    dependencies = payload.get("dependencies")
    local_imports = payload.get("local_imports")
    runtime_imports = payload.get("runtime_imports")
    if (
        payload.get("format") != R14_PYTHON_PROBE_FORMAT
        or payload.get("status")
        != "full_import_only_closure_verified_without_hdf5_access"
        or payload.get("required_imports") != list(R14_REQUIRED_PYTHON_IMPORTS)
        or payload.get("import_contract_sha256")
        != import_contract.get("contract_sha256")
        or payload.get("simulator_or_policy_objects_instantiated") != 0
        or payload.get("environment_reset_calls") != 0
        or payload.get("environment_step_calls") != 0
        or payload.get("output_roots_created") != 0
        or payload.get("environment_projection")
        != dict(environment_projection)
        or payload.get("hdf5_open_attempts") != 0
        or payload.get("hdf5_payloads_opened") != 0
        or payload.get("test_or_label_payloads_opened") != 0
        or not isinstance(interpreter, _R14Mapping)
        or interpreter.get("resolved_path") != python.get("resolved_path")
        or interpreter.get("sha256") != python.get("resolved_sha256")
        or not isinstance(interpreter.get("version"), str)
        or not interpreter.get("version")
        or not isinstance(interpreter.get("prefix"), str)
        or not interpreter.get("prefix")
        or not isinstance(interpreter.get("base_prefix"), str)
        or not interpreter.get("base_prefix")
        or not isinstance(interpreter.get("sys_path"), list)
        or not isinstance(dependencies, list)
        or len(dependencies) != len(R14_REQUIRED_PYTHON_IMPORTS)
        or not isinstance(local_imports, list)
        or len(local_imports) != len(R14_LOCAL_IMPORT_TARGETS)
        or not isinstance(runtime_imports, list)
        or len(runtime_imports) != len(R14_RUNTIME_IMPORT_TARGETS)
    ):
        raise WatcherContractError(
            "r14 Python dependency probe returned an invalid environment contract"
        )
    for expected_name, dependency in zip(
        R14_REQUIRED_PYTHON_IMPORTS, dependencies, strict=True
    ):
        origin = (
            dependency.get("origin")
            if isinstance(dependency, _R14Mapping)
            else None
        )
        distributions = (
            dependency.get("distributions")
            if isinstance(dependency, _R14Mapping)
            else None
        )
        if (
            not isinstance(dependency, _R14Mapping)
            or dependency.get("import_name") != expected_name
            or not isinstance(dependency.get("module_version"), str)
            or not dependency.get("module_version")
            or not isinstance(origin, _R14Mapping)
            or not _is_sha256(origin.get("sha256"))
            or not isinstance(origin.get("resolved_path"), str)
            or not origin.get("resolved_path")
            or not isinstance(origin.get("size"), int)
            or origin.get("size") <= 0
            or not isinstance(distributions, list)
        ):
            raise WatcherContractError(
                f"r14 Python dependency audit is incomplete: {expected_name}"
            )
        required_version = R14_REQUIRED_PYTHON_VERSIONS.get(expected_name)
        if (
            required_version is not None
            and dependency.get("module_version") != required_version
        ):
            raise WatcherContractError(
                f"r14 Python dependency version is invalid: {expected_name}"
            )
        origin_path = _R14Path(str(origin["resolved_path"])).resolve(strict=True)
        invocation_environment_root = _R14Path(
            str(python["invocation_path"])
        ).parent.parent.resolve(strict=True)
        prefix_root = _R14Path(str(interpreter["prefix"])).resolve(strict=True)
        base_prefix_root = _R14Path(
            str(interpreter["base_prefix"])
        ).resolve(strict=True)
        pyvenv = interpreter.get("pyvenv_config")
        if not isinstance(pyvenv, _R14Mapping):
            raise WatcherContractError("r14 Python pyvenv.cfg audit is missing")
        include_system = pyvenv.get("include_system_site_packages") is True
        if pyvenv.get("present") is True:
            content = pyvenv.get("content_utf8")
            parsed = pyvenv.get("parsed_fields")
            expected_config = (prefix_root / "pyvenv.cfg").resolve(strict=True)
            if (
                not isinstance(content, str)
                or not isinstance(parsed, _R14Mapping)
                or pyvenv.get("resolved_path") != str(expected_config)
                or pyvenv.get("sha256")
                != _r14_hashlib.sha256(content.encode("utf-8")).hexdigest()
                or pyvenv.get("size") != len(content.encode("utf-8"))
                or parsed.get("include_system_site_packages")
                not in {"true", "false"}
                or include_system
                != (
                    parsed.get("include_system_site_packages") == "true"
                )
            ):
                raise WatcherContractError("r14 Python pyvenv.cfg audit is invalid")
        elif (
            pyvenv
            != {
                "present": False,
                "include_system_site_packages": False,
            }
            or prefix_root != base_prefix_root
        ):
            raise WatcherContractError("r14 Python pyvenv.cfg binding is invalid")
        allowed_roots = [invocation_environment_root, prefix_root]
        if include_system:
            allowed_roots.append(base_prefix_root)
        if not any(
            root == origin_path or root in origin_path.parents
            for root in allowed_roots
        ):
            raise WatcherContractError(
                "r14 dependency escaped the selected Python environment: "
                f"{expected_name}"
            )
        for distribution in distributions:
            if (
                not isinstance(distribution, _R14Mapping)
                or not isinstance(distribution.get("name"), str)
                or not distribution.get("name")
                or not isinstance(distribution.get("version"), str)
                or not distribution.get("version")
                or distribution.get("metadata_sha256") is not None
                and not _is_sha256(distribution.get("metadata_sha256"))
                or distribution.get("record_sha256") is not None
                and not _is_sha256(distribution.get("record_sha256"))
            ):
                raise WatcherContractError(
                    f"r14 distribution audit is incomplete: {expected_name}"
                )
    for observed, expected in zip(
        local_imports, import_contract["local_imports"], strict=True
    ):
        origin = observed.get("origin") if isinstance(observed, _R14Mapping) else None
        if (
            not isinstance(observed, _R14Mapping)
            or observed.get("import_name") != expected.get("import_name")
            or observed.get("relative_path") != expected.get("relative_path")
            or not isinstance(origin, _R14Mapping)
            or origin.get("resolved_path") != expected.get("path")
            or origin.get("sha256") != expected.get("sha256")
            or origin.get("size") != expected.get("size")
        ):
            raise WatcherContractError("r14 local import-only closure changed")
    for observed, expected in zip(
        runtime_imports, import_contract["runtime_imports"], strict=True
    ):
        origin = observed.get("origin") if isinstance(observed, _R14Mapping) else None
        if (
            not isinstance(observed, _R14Mapping)
            or observed.get("import_name") != expected.get("import_name")
            or observed.get("artifact_role") != expected.get("artifact_role")
            or observed.get("symbol") != expected.get("symbol")
            or observed.get("symbol_present") is not True
            or not isinstance(origin, _R14Mapping)
            or origin.get("resolved_path") != expected.get("path")
            or origin.get("sha256") != expected.get("sha256")
            or origin.get("size") != expected.get("size")
        ):
            raise WatcherContractError("r14 signed runtime import closure changed")
    return dict(payload)


def _validate_r14_implementation_pins(
    plan: _R14Mapping[str, _R14Any],
) -> None:
    implementations = plan.get("implementation_files")
    if not isinstance(implementations, _R14Mapping):
        raise WatcherContractError("r14 schema6 implementation closure is missing")
    for relative_path, expected_sha256 in (
        R14_REQUIRED_IMPLEMENTATION_SHA256.items()
    ):
        record = implementations.get(relative_path)
        if (
            not isinstance(record, _R14Mapping)
            or record.get("sha256") != expected_sha256
        ):
            raise WatcherContractError(
                f"r14 requires the clean committed implementation: {relative_path}"
            )


def _run_python_dependency_probe(
    *,
    python: _R14Mapping[str, _R14Any],
    import_contract: _R14Mapping[str, _R14Any],
    gpu_index: int,
    omp_threads: int,
    timeout_seconds: float,
    base_environment: _R14Mapping[str, str] | None = None,
) -> dict[str, _R14Any]:
    if not 0 < timeout_seconds <= 600:
        raise WatcherContractError("r14 Python probe timeout is invalid")
    environment, _base_audit = isolated_stage_environment(
        _r14_os.environ if base_environment is None else base_environment,
        gpu_index=gpu_index,
        omp_threads=omp_threads,
    )
    runtime_roots = import_contract.get("runtime_roots")
    if not isinstance(runtime_roots, _R14Mapping):
        raise WatcherContractError("r14 import-only runtime roots are missing")
    environment.update(
        {
            "ASSETS_PATH": str(runtime_roots["robotwin_root"]),
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    projection = _effective_environment_projection(environment)
    argv = [
        str(python["invocation_path"]),
        "-I",
        "-B",
        "-c",
        _R14_PYTHON_PROBE_SOURCE,
        _r14_json.dumps(list(R14_REQUIRED_PYTHON_IMPORTS), separators=(",", ":")),
        _r14_json.dumps(list(R14_AUDITED_ENVIRONMENT_NAMES), separators=(",", ":")),
        _r14_json.dumps(
            dict(import_contract), sort_keys=True, separators=(",", ":")
        ),
    ]
    try:
        completed = _r14_subprocess.run(
            argv,
            stdin=_r14_subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            close_fds=True,
            env=environment,
            cwd=str(_R14Path(str(python["resolved_path"])).parent),
            timeout=timeout_seconds,
        )
    except (
        OSError,
        _r14_subprocess.SubprocessError,
    ) as error:
        raise WatcherContractError(
            "r14 Python dependency probe could not complete"
        ) from error
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or completed.stderr or len(stdout_lines) != 1:
        raise WatcherContractError(
            "r14 Python environment cannot import all required dependencies cleanly"
        )
    try:
        payload = _r14_json.loads(stdout_lines[0])
    except _r14_json.JSONDecodeError as error:
        raise WatcherContractError(
            "r14 Python dependency probe did not emit canonical JSON"
        ) from error
    if not isinstance(payload, _R14Mapping):
        raise WatcherContractError(
            "r14 Python dependency probe payload is not an object"
        )
    validated = _validate_probe_payload(
        payload,
        python=python,
        environment_projection=projection,
        import_contract=import_contract,
    )
    audit: dict[str, _R14Any] = {
        "format": R14_PYTHON_ENVIRONMENT_FORMAT,
        "status": "python_interpreter_environment_dependencies_verified",
        "python_contract": dict(python),
        "probe_source_sha256": R14_PYTHON_PROBE_SOURCE_SHA256,
        "probe_argv_sha256": canonical_sha256(
            [
                argv[0],
                "-I",
                "-B",
                "-c",
                R14_PYTHON_PROBE_SOURCE_SHA256,
                argv[-3],
                argv[-2],
                import_contract["contract_sha256"],
            ]
        ),
        "required_imports": list(R14_REQUIRED_PYTHON_IMPORTS),
        "import_contract": dict(import_contract),
        "import_contract_sha256": import_contract["contract_sha256"],
        "environment_projection": projection,
        "probe_payload": validated,
        "hdf5_payloads_opened": 0,
        "test_or_label_payloads_opened": 0,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def _validate_r14_output_root(output: _R14Path) -> None:
    supplied = reject_path_text(output, "r14 schema6 output root")
    if supplied in R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS:
        raise WatcherContractError("r14 cannot reuse the frozen r13 schema6 output")
    if supplied != R14_SCHEMA6_OUTPUT_ROOT:
        raise WatcherContractError(
            "r14 schema6 output differs from its designated new output root"
        )


def _validate_r14_plan_binding(plan: _R14Mapping[str, _R14Any]) -> None:
    _validate_r14_implementation_pins(plan)
    binding = plan.get("r14_static_binding")
    python_audit = plan.get("python_environment_audit")
    import_contract = plan.get("import_only_closure")
    if (
        binding != r14_static_binding()
        or plan.get("r14_static_binding_sha256") != R14_STATIC_BINDING_SHA256
        or plan.get("r14_static_binding_sha256")
        != canonical_sha256(r14_static_binding())
        or not isinstance(python_audit, _R14Mapping)
        or python_audit.get("format") != R14_PYTHON_ENVIRONMENT_FORMAT
        or python_audit.get("status")
        != "python_interpreter_environment_dependencies_verified"
        or python_audit.get("audit_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in python_audit.items()
                if key != "audit_sha256"
            }
        )
        or plan.get("python_environment_audit_sha256")
        != python_audit.get("audit_sha256")
        or not isinstance(import_contract, _R14Mapping)
        or import_contract.get("format") != R14_IMPORT_CONTRACT_FORMAT
        or import_contract.get("contract_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in import_contract.items()
                if key != "contract_sha256"
            }
        )
        or plan.get("import_only_closure_sha256")
        != import_contract.get("contract_sha256")
        or python_audit.get("import_contract") != import_contract
        or python_audit.get("import_contract_sha256")
        != import_contract.get("contract_sha256")
        or plan.get("code_root") != str(R14_CODE_ROOT)
        or plan.get("output_root") != str(R14_SCHEMA6_OUTPUT_ROOT)
        or not isinstance(plan.get("omp_threads"), int)
        or plan.get("omp_threads") <= 0
        or python_audit.get("environment_projection", {}).get(
            "OMP_NUM_THREADS"
        )
        != str(plan.get("omp_threads"))
        or python_audit.get("environment_projection", {}).get(
            "CUDA_VISIBLE_DEVICES"
        )
        != str(plan.get("gpu_index"))
        or plan.get("prior_code_root_reused") is not False
        or plan.get("prior_schema6_output_reused") is not False
        or plan.get("hdf5_payloads_opened_during_python_preflight") != 0
        or plan.get("test_or_label_payloads_opened_during_python_preflight") != 0
    ):
        raise WatcherContractError(
            "schema6 static plan lost its r14 Python environment binding"
        )


def static_preflight(args: argparse.Namespace) -> dict[str, _R14Any]:
    _verify_r14_parent_implementation()
    _validate_r14_code_root(args.code_root)
    _validate_r14_output_root(args.output)
    requested_python = reject_path_text(args.python_bin, "r14 Python executable")
    previous_designated_python = DESIGNATED_PYTHON
    previous_designated_code_root = DESIGNATED_CODE_ROOT
    globals()["DESIGNATED_PYTHON"] = requested_python
    globals()["DESIGNATED_CODE_ROOT"] = R14_CODE_ROOT
    try:
        plan = dict(_r14_parent_static_preflight(args))
    finally:
        globals()["DESIGNATED_PYTHON"] = previous_designated_python
        globals()["DESIGNATED_CODE_ROOT"] = previous_designated_code_root
    _validate_r14_implementation_pins(plan)
    import_contract = _build_r14_import_contract(plan)
    python_audit = _run_python_dependency_probe(
        python=plan["python_contract"],
        import_contract=import_contract,
        gpu_index=int(plan["gpu_index"]),
        omp_threads=int(plan["subprocess_environment_contract"][
            "forced_python_environment"
        ].get("OMP_NUM_THREADS", args.omp_threads)),
        timeout_seconds=float(args.python_probe_timeout_seconds),
    )
    plan.pop("static_plan_sha256", None)
    plan.update(
        {
            "status": (
                "static_preflight_complete_python_environment_verified_"
                "lobo_summary_not_read"
            ),
            "r14_static_binding": r14_static_binding(),
            "r14_static_binding_sha256": R14_STATIC_BINDING_SHA256,
            "import_only_closure": import_contract,
            "import_only_closure_sha256": import_contract["contract_sha256"],
            "python_environment_audit": python_audit,
            "python_environment_audit_sha256": python_audit["audit_sha256"],
            "python_probe_timeout_seconds": float(
                args.python_probe_timeout_seconds
            ),
            "omp_threads": int(args.omp_threads),
            "prior_code_root_reused": False,
            "prior_schema6_output_reused": False,
            "hdf5_payloads_opened_during_python_preflight": 0,
            "test_or_label_payloads_opened_during_python_preflight": 0,
        }
    )
    plan["static_plan_sha256"] = canonical_sha256(plan)
    _validate_r14_plan_binding(plan)
    return plan


def verify_static_bindings(plan: _R14Mapping[str, _R14Any]) -> None:
    _verify_r14_parent_implementation()
    _validate_r14_plan_binding(plan)
    _r14_parent_verify_static_bindings(plan)
    import_contract = _build_r14_import_contract(plan)
    if import_contract != plan["import_only_closure"]:
        raise WatcherContractError("r14 signed import-only closure changed")
    current = _run_python_dependency_probe(
        python=plan["python_contract"],
        import_contract=import_contract,
        gpu_index=int(plan["gpu_index"]),
        omp_threads=int(plan["omp_threads"]),
        timeout_seconds=float(plan["python_probe_timeout_seconds"]),
    )
    if current != plan["python_environment_audit"]:
        raise WatcherContractError(
            "r14 Python interpreter or dependency environment changed"
        )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    _r14_parent_add_common_arguments(parser)
    parser.add_argument(
        "--python-probe-timeout-seconds",
        type=float,
        default=R14_DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS,
    )


def _python_bin_was_explicit(argv: _R14Sequence[str]) -> bool:
    return any(
        item == "--python-bin" or item.startswith("--python-bin=")
        for item in argv
    )


def parse_args() -> argparse.Namespace:
    args = _r14_parent_parse_args()
    if not _python_bin_was_explicit(_r14_sys.argv[1:]):
        raise SystemExit("r14 requires an explicit --python-bin binding")
    if not 0 < args.python_probe_timeout_seconds <= 600:
        raise SystemExit("r14 Python dependency probe timeout must be in (0, 600]")
    return args


def _common_run_argv(args: argparse.Namespace) -> list[str]:
    argv = _r14_parent_common_run_argv(args)
    argv.extend(
        [
            "--python-probe-timeout-seconds",
            str(args.python_probe_timeout_seconds),
        ]
    )
    return argv


__all__ = list(globals().get("__all__", [])) + [
    "R14_AUDITED_ENVIRONMENT_NAMES",
    "R14_BINDING_FORMAT",
    "R14_CODE_ROOT",
    "R14_FORBIDDEN_PRIOR_CODE_ROOTS",
    "R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS",
    "R14_IMPORT_CONTRACT_FORMAT",
    "R14_LOCAL_IMPORT_TARGETS",
    "R14_PARENT_IMPLEMENTATION_FILENAME",
    "R14_PARENT_IMPLEMENTATION_SHA256",
    "R14_PYTHON_ENVIRONMENT_FORMAT",
    "R14_PYTHON_PROBE_SOURCE_SHA256",
    "R14_REQUIRED_PYTHON_IMPORTS",
    "R14_REQUIRED_PYTHON_VERSIONS",
    "R14_REQUIRED_IMPLEMENTATION_SHA256",
    "R14_RUNTIME_IMPORT_TARGETS",
    "R14_SCHEMA6_OUTPUT_ROOT",
    "R14_STATIC_BINDING_SHA256",
    "r14_static_binding",
]


if _r14_public_module_name == "__main__":
    main()
