from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_smolvla_piper_schema6_autonomous_watcher_r14 as watcher  # noqa: E402


def _import_fixture(tmp_path: Path) -> dict:
    roots = {
        name: tmp_path / name
        for name in ("robotwin_root", "robotwin_code", "rlinf_root", "lerobot_root")
    }
    for root in roots.values():
        root.mkdir(parents=True)
    local_imports = []
    for target in watcher.R14_LOCAL_IMPORT_TARGETS:
        path = tmp_path / "code" / target["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = target["import_name"].encode()
        path.write_bytes(raw)
        local_imports.append(
            {
                **dict(target),
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    runtime_imports = []
    for target in watcher.R14_RUNTIME_IMPORT_TARGETS:
        path = tmp_path / "runtime" / f"{target['artifact_role']}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = target["import_name"].encode()
        path.write_bytes(raw)
        runtime_imports.append(
            {
                **dict(target),
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    value = {
        "format": watcher.R14_IMPORT_CONTRACT_FORMAT,
        "code_root": str((tmp_path / "code").resolve()),
        "r6f_preregistration_sha256": "d" * 64,
        "runtime_roots": {
            name: str(root.resolve()) for name, root in roots.items()
        },
        "python_paths": [
            str((tmp_path / "code" / "scripts").resolve()),
            str(roots["robotwin_code"].resolve()),
            str(roots["rlinf_root"].resolve()),
            str(roots["lerobot_root"].resolve() / "src"),
        ],
        "local_imports": local_imports,
        "runtime_imports": runtime_imports,
        "import_only": True,
        "simulator_or_policy_objects_instantiated": 0,
        "environment_reset_calls": 0,
        "environment_step_calls": 0,
        "hdf5_payloads_opened": 0,
        "test_or_label_payloads_opened": 0,
    }
    value["contract_sha256"] = watcher.canonical_sha256(value)
    return value


def _python_fixture(
    tmp_path: Path,
    *,
    system_site_imports: frozenset[str] = frozenset(),
) -> tuple[dict, dict, dict, dict]:
    import_contract = _import_fixture(tmp_path)
    environment_root = tmp_path / "selected_venv"
    base_prefix = tmp_path / "base_python" if system_site_imports else environment_root
    base_prefix.mkdir(parents=True, exist_ok=True)
    python = environment_root / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"audited-python")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    site = environment_root / "lib" / "python3.10" / "site-packages"
    dependencies = []
    for name in watcher.R14_REQUIRED_PYTHON_IMPORTS:
        origin_root = (
            base_prefix / "lib" / "python3.10" / "site-packages"
            if name in system_site_imports
            else site
        )
        origin = origin_root / name / "__init__.py"
        origin.parent.mkdir(parents=True)
        raw = f"{name}-module".encode()
        origin.write_bytes(raw)
        version = watcher.R14_REQUIRED_PYTHON_VERSIONS.get(name, "1.0")
        distributions = []
        if name in watcher.R14_REQUIRED_PYTHON_VERSIONS:
            distributions.append(
                {
                    "name": (
                        "antlr4-python3-runtime" if name == "antlr4" else name
                    ),
                    "version": version,
                    "metadata_sha256": "a" * 64,
                    "record_sha256": "b" * 64,
                }
            )
        dependencies.append(
            {
                "import_name": name,
                "module_version": version,
                "origin": {
                    "invocation_path": str(origin),
                    "resolved_path": str(origin.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                },
                "distributions": distributions,
            }
        )
    contract = {
        "invocation_path": str(python),
        "resolved_path": str(python.resolve()),
        "resolved_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
    }
    if system_site_imports:
        pyvenv_content = "home = /usr/bin\ninclude-system-site-packages = true\n"
        pyvenv_path = environment_root / "pyvenv.cfg"
        pyvenv_path.write_text(pyvenv_content, encoding="utf-8")
        pyvenv = {
            "invocation_path": str(pyvenv_path),
            "resolved_path": str(pyvenv_path.resolve()),
            "sha256": hashlib.sha256(pyvenv_content.encode()).hexdigest(),
            "size": len(pyvenv_content.encode()),
            "present": True,
            "content_utf8": pyvenv_content,
            "parsed_fields": {
                "home": "/usr/bin",
                "include_system_site_packages": "true",
            },
            "include_system_site_packages": True,
        }
    else:
        pyvenv = {
            "present": False,
            "include_system_site_packages": False,
        }
    projection = {
        name: None for name in watcher.R14_AUDITED_ENVIRONMENT_NAMES
    }
    projection.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "ASSETS_PATH": import_contract["runtime_roots"]["robotwin_root"],
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    payload = {
        "format": watcher.R14_PYTHON_PROBE_FORMAT,
        "status": "full_import_only_closure_verified_without_hdf5_access",
        "interpreter": {
            "invocation_path": str(python),
            "resolved_path": str(python.resolve()),
            "sha256": contract["resolved_sha256"],
            "size": python.stat().st_size,
            "version": "3.10 synthetic",
            "implementation": "cpython",
            "cache_tag": "cpython-310",
            "prefix": str(environment_root),
            "base_prefix": str(base_prefix),
            "pyvenv_config": pyvenv,
            "sys_path": [str(site)],
        },
        "required_imports": list(watcher.R14_REQUIRED_PYTHON_IMPORTS),
        "dependencies": dependencies,
        "import_contract_sha256": import_contract["contract_sha256"],
        "local_imports": [
            {
                "import_name": value["import_name"],
                "relative_path": value["relative_path"],
                "origin": {
                    "invocation_path": value["path"],
                    "resolved_path": value["path"],
                    "sha256": value["sha256"],
                    "size": value["size"],
                },
            }
            for value in import_contract["local_imports"]
        ],
        "runtime_imports": [
            {
                "import_name": value["import_name"],
                "artifact_role": value["artifact_role"],
                "symbol": value["symbol"],
                "symbol_present": True,
                "origin": {
                    "invocation_path": value["path"],
                    "resolved_path": value["path"],
                    "sha256": value["sha256"],
                    "size": value["size"],
                },
            }
            for value in import_contract["runtime_imports"]
        ],
        "simulator_or_policy_objects_instantiated": 0,
        "environment_reset_calls": 0,
        "environment_step_calls": 0,
        "environment_projection": projection,
        "hdf5_open_attempts": 0,
        "hdf5_payloads_opened": 0,
        "test_or_label_payloads_opened": 0,
        "output_roots_created": 0,
    }
    return contract, projection, payload, import_contract


def _environment_audit(
    contract: dict, projection: dict, payload: dict, import_contract: dict
) -> dict:
    value = {
        "format": watcher.R14_PYTHON_ENVIRONMENT_FORMAT,
        "status": "python_interpreter_environment_dependencies_verified",
        "python_contract": contract,
        "probe_source_sha256": watcher.R14_PYTHON_PROBE_SOURCE_SHA256,
        "probe_argv_sha256": "c" * 64,
        "required_imports": list(watcher.R14_REQUIRED_PYTHON_IMPORTS),
        "import_contract": import_contract,
        "import_contract_sha256": import_contract["contract_sha256"],
        "environment_projection": projection,
        "probe_payload": payload,
        "hdf5_payloads_opened": 0,
        "test_or_label_payloads_opened": 0,
    }
    value["audit_sha256"] = watcher.canonical_sha256(value)
    return value


def _base_plan(tmp_path: Path, contract: dict) -> dict:
    return {
        "format": watcher.FORMAT,
        "status": "base_preflight",
        "output_root": str(watcher.R14_SCHEMA6_OUTPUT_ROOT),
        "code_root": str(watcher.R14_CODE_ROOT),
        "implementation_files": {
            path: {"path": str(tmp_path / Path(path).name), "sha256": digest}
            for path, digest in watcher.R14_REQUIRED_IMPLEMENTATION_SHA256.items()
        },
        "python_contract": contract,
        "gpu_index": 0,
        "subprocess_environment_contract": {
            "forced_python_environment": {}
        },
        "static_plan_sha256": "stale",
    }


def test_r14_constants_pin_parent_clean_collect_and_new_output() -> None:
    assert watcher.R14_PARENT_IMPLEMENTATION_SHA256 == (
        "dc548a5a8155dfd479da521f41c033417c3bfb260011f2f54865282fd1952da1"
    )
    assert watcher.R14_REQUIRED_IMPLEMENTATION_SHA256 == {
        "scripts/launch_smolvla_piper_schema6_development_collection.py": (
            "6903fb0d44c49878bfb56c05c463a264640cf8c12f000cb7f068a5cba8037c4e"
        ),
        "scripts/materialize_smolvla_piper_schema6_reset_contract.py": (
            "03f05d5510bd40a7a71dd9e69db924c1aea50ea185ae2a84f60bde08f38dcf1b"
        ),
        "scripts/collect_openvla_etsf_rollouts.py": (
            "b8a20dcf15dea31d7708cf90c208b260d410eabd681499ab6101e6cb3cf8d491"
        )
    }
    assert watcher.R14_REQUIRED_PYTHON_VERSIONS == {
        "omegaconf": "2.3.0",
        "antlr4": "4.9.3",
    }
    assert watcher.R14_SCHEMA6_OUTPUT_ROOT not in (
        watcher.R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS
    )
    assert watcher.R14_CODE_ROOT == Path(
        "/home/user/etsf_smolvla_piper_schema6_code_r14d_20260829"
    )
    assert watcher.R14_CODE_ROOT not in watcher.R14_FORBIDDEN_PRIOR_CODE_ROOTS
    assert {value["import_name"] for value in watcher.R14_LOCAL_IMPORT_TARGETS} >= {
        "materialize_smolvla_piper_schema6_reset_contract",
        "launch_smolvla_piper_schema6_development_collection",
        "collect_openvla_etsf_rollouts",
    }
    assert {value["symbol"] for value in watcher.R14_RUNTIME_IMPORT_TARGETS} >= {
        "RoboTwinEnv",
        "make_pre_post_processors",
        "SmolVLAPolicy",
    }


def test_probe_uses_isolated_no_user_site_environment_and_validates_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, projection, payload, import_contract = _python_fixture(tmp_path)

    def fake_run(argv, **kwargs):
        assert argv[1:3] == ["-I", "-B"]
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["stdin"] is watcher._r14_subprocess.DEVNULL
        emitted = dict(payload)
        emitted["environment_projection"] = {
            name: kwargs["env"].get(name)
            for name in watcher.R14_AUDITED_ENVIRONMENT_NAMES
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(emitted, sort_keys=True) + "\n",
            stderr="",
        )

    monkeypatch.setattr(watcher._r14_subprocess, "run", fake_run)
    audit = watcher._run_python_dependency_probe(
        python=contract,
        import_contract=import_contract,
        gpu_index=0,
        omp_threads=8,
        timeout_seconds=10,
        base_environment={},
    )
    assert audit["environment_projection"] == projection
    assert audit["hdf5_payloads_opened"] == 0
    assert audit["test_or_label_payloads_opened"] == 0


def test_probe_binds_environment_before_import_side_effects() -> None:
    source = watcher._R14_PYTHON_PROBE_SOURCE
    snapshot = "launch_environment_projection = {"
    first_import = "packages_to_distributions = importlib.metadata.packages_distributions()"
    assert source.index(snapshot) < source.index(first_import)
    assert '"environment_projection": launch_environment_projection' in source


def test_probe_rejects_wrong_omegaconf_version(tmp_path: Path) -> None:
    contract, projection, payload, import_contract = _python_fixture(tmp_path)
    payload["dependencies"][2]["module_version"] = "2.2.0"
    with pytest.raises(
        watcher.WatcherContractError,
        match="dependency version is invalid: omegaconf",
    ):
        watcher._validate_probe_payload(
            payload,
            python=contract,
            environment_projection=projection,
            import_contract=import_contract,
        )


def test_probe_rejects_user_site_dependency_origin(tmp_path: Path) -> None:
    contract, projection, payload, import_contract = _python_fixture(tmp_path)
    outside = tmp_path / "user_site" / "omegaconf" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"user-site")
    payload["dependencies"][2]["origin"].update(
        {
            "resolved_path": str(outside.resolve()),
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
            "size": outside.stat().st_size,
        }
    )
    with pytest.raises(
        watcher.WatcherContractError,
        match="escaped the selected Python environment: omegaconf",
    ):
        watcher._validate_probe_payload(
            payload,
            python=contract,
            environment_projection=projection,
            import_contract=import_contract,
        )


def test_probe_accepts_bound_system_site_packages_from_base_prefix(
    tmp_path: Path,
) -> None:
    contract, projection, payload, import_contract = _python_fixture(
        tmp_path, system_site_imports=frozenset({"torch", "yaml"})
    )
    validated = watcher._validate_probe_payload(
        payload,
        python=contract,
        environment_projection=projection,
        import_contract=import_contract,
    )
    assert validated["interpreter"]["pyvenv_config"][
        "include_system_site_packages"
    ] is True


def test_probe_rejects_base_prefix_without_bound_pyvenv_config(
    tmp_path: Path,
) -> None:
    contract, projection, payload, import_contract = _python_fixture(
        tmp_path, system_site_imports=frozenset({"torch"})
    )
    payload["interpreter"]["pyvenv_config"] = {
        "present": False,
        "include_system_site_packages": False,
    }
    with pytest.raises(watcher.WatcherContractError, match="pyvenv.cfg binding"):
        watcher._validate_probe_payload(
            payload,
            python=contract,
            environment_projection=projection,
            import_contract=import_contract,
        )


def test_probe_rejects_tampered_local_or_runtime_import(
    tmp_path: Path,
) -> None:
    contract, projection, payload, import_contract = _python_fixture(tmp_path)
    payload["local_imports"][2]["origin"]["sha256"] = "0" * 64
    with pytest.raises(watcher.WatcherContractError, match="local import-only"):
        watcher._validate_probe_payload(
            payload,
            python=contract,
            environment_projection=projection,
            import_contract=import_contract,
        )
    payload["local_imports"][2]["origin"]["sha256"] = import_contract[
        "local_imports"
    ][2]["sha256"]
    payload["runtime_imports"][0]["origin"]["sha256"] = "0" * 64
    with pytest.raises(watcher.WatcherContractError, match="runtime import closure"):
        watcher._validate_probe_payload(
            payload,
            python=contract,
            environment_projection=projection,
            import_contract=import_contract,
        )


def test_signed_runtime_source_accepts_writable_identical_bytes_and_rejects_change(
    tmp_path: Path,
) -> None:
    source = tmp_path / "shared_runtime.py"
    source.write_bytes(b"signed-runtime")
    source.chmod(0o664)
    signed = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    record = watcher._bound_source_record(signed, role="shared_runtime")
    assert source.stat().st_mode & 0o222
    assert record["size"] == len(b"signed-runtime")
    assert record["read_consistency"].startswith("same_device_inode")
    source.write_bytes(b"changed-runtime")
    with pytest.raises(watcher.WatcherContractError, match="source changed"):
        watcher._bound_source_record(signed, role="shared_runtime")


def test_preflight_rejects_frozen_r6j_code_root_before_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"parent": False}
    monkeypatch.setattr(
        watcher,
        "_r14_parent_static_preflight",
        lambda _args: called.update(parent=True),
    )
    with pytest.raises(watcher.WatcherContractError, match="cannot reuse"):
        watcher.static_preflight(
            argparse.Namespace(
                code_root=watcher.R14_FORBIDDEN_PRIOR_CODE_ROOTS[0],
                output=watcher.R14_SCHEMA6_OUTPUT_ROOT,
                python_bin=Path(sys.executable),
            )
        )
    assert called["parent"] is False


def test_preflight_rejects_r13_output_before_parent_or_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = tmp_path / "schema6_r13"
    monkeypatch.setattr(
        watcher, "R14_FORBIDDEN_PRIOR_SCHEMA6_OUTPUT_ROOTS", (previous,)
    )
    called = {"parent": False}
    monkeypatch.setattr(
        watcher,
        "_r14_parent_static_preflight",
        lambda _args: called.update(parent=True),
    )
    with pytest.raises(watcher.WatcherContractError, match="cannot reuse"):
        watcher.static_preflight(
            argparse.Namespace(
                code_root=watcher.R14_CODE_ROOT,
                output=previous,
                python_bin=Path(sys.executable),
            )
        )
    assert called["parent"] is False
    assert not previous.exists()


def test_preflight_fails_before_output_when_dependency_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, _projection, _payload, import_contract = _python_fixture(tmp_path)
    output = tmp_path / "schema6_r14"
    monkeypatch.setattr(watcher, "R14_SCHEMA6_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        watcher,
        "R14_STATIC_BINDING_SHA256",
        watcher.canonical_sha256(watcher.r14_static_binding()),
    )
    monkeypatch.setattr(
        watcher,
        "_r14_parent_static_preflight",
        lambda _args: _base_plan(tmp_path, contract),
    )
    monkeypatch.setattr(
        watcher, "_build_r14_import_contract", lambda _plan: import_contract
    )
    monkeypatch.setattr(
        watcher,
        "_run_python_dependency_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            watcher.WatcherContractError("omegaconf missing")
        ),
    )
    args = argparse.Namespace(
        output=output,
        code_root=watcher.R14_CODE_ROOT,
        python_bin=Path(contract["invocation_path"]),
        omp_threads=8,
        python_probe_timeout_seconds=10,
    )
    with pytest.raises(watcher.WatcherContractError, match="omegaconf missing"):
        watcher.static_preflight(args)
    assert not output.exists()


def test_preflight_binds_probe_and_verify_reprobes_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, projection, payload, import_contract = _python_fixture(tmp_path)
    audit = _environment_audit(contract, projection, payload, import_contract)
    output = tmp_path / "schema6_r14"
    monkeypatch.setattr(watcher, "R14_SCHEMA6_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        watcher,
        "R14_STATIC_BINDING_SHA256",
        watcher.canonical_sha256(watcher.r14_static_binding()),
    )
    previous_code_root = watcher.DESIGNATED_CODE_ROOT

    def fake_parent(_args):
        assert watcher.DESIGNATED_CODE_ROOT == watcher.R14_CODE_ROOT
        assert watcher.DESIGNATED_PYTHON == Path(contract["invocation_path"])
        return _base_plan(tmp_path, contract)

    monkeypatch.setattr(watcher, "_r14_parent_static_preflight", fake_parent)
    monkeypatch.setattr(
        watcher, "_build_r14_import_contract", lambda _plan: import_contract
    )
    probes = []
    monkeypatch.setattr(
        watcher,
        "_run_python_dependency_probe",
        lambda **_kwargs: probes.append(True) or dict(audit),
    )
    args = argparse.Namespace(
        output=output,
        code_root=watcher.R14_CODE_ROOT,
        python_bin=Path(contract["invocation_path"]),
        omp_threads=8,
        python_probe_timeout_seconds=10,
    )
    plan = watcher.static_preflight(args)
    assert watcher.DESIGNATED_CODE_ROOT == previous_code_root
    assert probes == [True]
    assert plan["python_environment_audit"] == audit
    assert plan["hdf5_payloads_opened_during_python_preflight"] == 0
    assert plan["test_or_label_payloads_opened_during_python_preflight"] == 0
    assert not output.exists()
    monkeypatch.setattr(
        watcher, "_r14_parent_verify_static_bindings", lambda _plan: None
    )
    watcher.verify_static_bindings(plan)
    assert probes == [True, True]


def test_dirty_collect_implementation_is_rejected() -> None:
    plan = {
        "implementation_files": {
            "scripts/launch_smolvla_piper_schema6_development_collection.py": {
                "sha256": watcher.R14_REQUIRED_IMPLEMENTATION_SHA256[
                    "scripts/launch_smolvla_piper_schema6_development_collection.py"
                ]
            },
            "scripts/materialize_smolvla_piper_schema6_reset_contract.py": {
                "sha256": watcher.R14_REQUIRED_IMPLEMENTATION_SHA256[
                    "scripts/materialize_smolvla_piper_schema6_reset_contract.py"
                ]
            },
            "scripts/collect_openvla_etsf_rollouts.py": {
                "sha256": "32f157" + "0" * 58
            }
        }
    }
    with pytest.raises(
        watcher.WatcherContractError,
        match="clean committed implementation",
    ):
        watcher._validate_r14_implementation_pins(plan)


def test_python_bin_must_be_explicit() -> None:
    assert watcher._python_bin_was_explicit(["preflight", "--python-bin", "/x"])
    assert watcher._python_bin_was_explicit(["preflight", "--python-bin=/x"])
    assert not watcher._python_bin_was_explicit(["preflight", "--output", "/x"])
