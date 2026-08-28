from __future__ import annotations

import copy
import hashlib
import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preregister_smolvla_piper_paired_execution_inventory_v3 as prereg  # noqa: E402
import smolvla_piper_paired_success_protocol_v3 as paired  # noqa: E402


@pytest.fixture
def safe_root(tmp_path: Path) -> Iterator[Path]:
    root = Path(tempfile.mkdtemp(prefix="inventory_v3_", dir=tmp_path.parent))
    try:
        yield root
    finally:
        for path in sorted(root.rglob("*"), reverse=True):
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except OSError:
                pass
        root.chmod(0o700)
        shutil.rmtree(root)


def frozen(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_manifest(root: Path) -> dict[str, Any]:
    prereg.generate_keys(root / "keys")
    return json.loads((root / "keys" / "public_key_manifest.json").read_text())


def case(root: Path, *, shared: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = key_manifest(root)
    component_root = root / "components"
    executor = frozen(component_root / "paired_executor_v3.py", b"# executor\n")
    evaluator = frozen(component_root / "paired_result_evaluator_v3.py", b"# evaluator\n")
    runner = frozen(component_root / "condition_runner.py", b"# condition runner\n")
    collector = runner if shared else frozen(
        component_root / "collector_runner.py", b"# collector runner\n"
    )
    runtime = frozen(component_root / "runtime_contract.json", b'{"runtime":"fixed"}\n')
    container = frozen(
        component_root / "container_inventory.json", b'{"image_digest":"sha256:fixed"}\n'
    )
    keys = manifest["keys"]
    kwargs = {
        "issuer_key_id": "external-evaluation-issuer-v3",
        "issuer_public_key": Path(keys["issuer"]["public_key_path"]),
        "issuer_public_key_file_sha256": keys["issuer"]["public_key_file_sha256"],
        "executor_public_key": Path(keys["executor"]["public_key_path"]),
        "executor_public_key_file_sha256": keys["executor"]["public_key_file_sha256"],
        "result_signer_public_key": Path(keys["result_signer"]["public_key_path"]),
        "result_signer_public_key_file_sha256": keys["result_signer"]["public_key_file_sha256"],
        "executor_implementation": executor,
        "executor_implementation_file_sha256": sha(executor),
        "result_evaluator_implementation": evaluator,
        "result_evaluator_implementation_file_sha256": sha(evaluator),
        "simulator_implementation": runner,
        "simulator_implementation_file_sha256": sha(runner),
        "collector_implementation": collector,
        "collector_implementation_file_sha256": sha(collector),
        "runtime_contract": runtime,
        "runtime_contract_file_sha256": sha(runtime),
        "container_inventory": container,
        "container_inventory_file_sha256": sha(container),
        "condition_runner_binding": "shared" if shared else "distinct",
        "issuer_attestation_output": root / "trusted_issuer_attestation.json",
        "inventory_output": root / "execution_inventory_attestation.json",
    }
    return kwargs, manifest


def test_generate_keys_creates_three_raw_private_keys_and_public_manifest(
    safe_root: Path,
) -> None:
    manifest = key_manifest(safe_root)
    assert manifest["format"] == prereg.KEY_MANIFEST_FORMAT
    assert manifest["execution_authorized"] is False
    assert type(manifest["private_key_count"]) is int
    assert set(manifest["keys"]) == {"issuer", "executor", "result_signer"}
    identities = set()
    for role, row in manifest["keys"].items():
        private_path = safe_root / "keys" / f"{role}_ed25519_private.raw"
        public_path = Path(row["public_key_path"])
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o400
        assert stat.S_IMODE(public_path.stat().st_mode) == 0o444
        assert len(private_path.read_bytes()) == 32
        assert len(public_path.read_bytes()) == 32
        assert "private_key_path" not in row
        assert row["identity_sha256"] == sha(public_path)
        identities.add(row["identity_sha256"])
    assert len(identities) == 3
    assert stat.S_IMODE((safe_root / "keys" / "public_key_manifest.json").stat().st_mode) == 0o444


def test_preregister_shared_runner_matches_paired_v3_validator(
    safe_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, manifest = case(safe_root, shared=True)
    issuer, inventory = prereg.preregister(**kwargs)
    inventory_path = kwargs["inventory_output"]
    inventory_file_sha = sha(inventory_path)
    assert inventory["executor"]["identity_sha256"] == manifest["keys"]["executor"][
        "identity_sha256"
    ]
    assert inventory["result_evaluator"]["identity_sha256"] == manifest["keys"][
        "result_signer"
    ]["identity_sha256"]
    assert (
        inventory["execution_stack"]["simulator_implementation"]
        == inventory["execution_stack"]["collector_implementation"]
    )
    assert issuer["issuer_public_key_sha256"] == manifest["keys"]["issuer"][
        "identity_sha256"
    ]
    monkeypatch.setattr(
        paired, "APPROVED_EXECUTION_INVENTORY_FILE_SHA256", inventory_file_sha
    )
    decoded, decoded_issuer, record, _stack_sha = paired._validate_execution_inventory(
        path=inventory_path,
        expected_file_sha256=inventory_file_sha,
        expected_logical_sha256=inventory["attestation_sha256"],
    )
    assert decoded == inventory
    assert decoded_issuer["attestation_sha256"] == issuer["attestation_sha256"]
    assert record["file_sha256"] == inventory_file_sha


def test_distinct_condition_runners_are_supported(safe_root: Path) -> None:
    kwargs, _manifest = case(safe_root, shared=False)
    _issuer, inventory = prereg.preregister(**kwargs)
    assert (
        inventory["execution_stack"]["simulator_implementation"]
        != inventory["execution_stack"]["collector_implementation"]
    )


def test_condition_runner_declaration_must_match_exact_descriptors(
    safe_root: Path,
) -> None:
    kwargs, _manifest = case(safe_root, shared=True)
    kwargs["condition_runner_binding"] = "distinct"
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="declaration"
    ):
        prereg.preregister(**kwargs)


def test_symlink_and_writable_components_fail_closed(safe_root: Path) -> None:
    kwargs, _manifest = case(safe_root, shared=True)
    executor = kwargs["executor_implementation"]
    alias = safe_root / "components" / "executor_alias.py"
    alias.symlink_to(executor)
    kwargs["executor_implementation"] = alias
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="symlinks"
    ):
        prereg.preregister(**kwargs)
    kwargs["executor_implementation"] = executor
    executor.chmod(0o644)
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="frozen read-only"
    ):
        prereg.preregister(**kwargs)


def test_duplicate_runtime_json_and_bool_count_fail_closed(safe_root: Path) -> None:
    kwargs, _manifest = case(safe_root, shared=True)
    runtime = kwargs["runtime_contract"]
    runtime.chmod(0o644)
    runtime.write_bytes(b'{"runtime":1,"runtime":2}\n')
    runtime.chmod(0o444)
    kwargs["runtime_contract_file_sha256"] = sha(runtime)
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="duplicate JSON key"
    ):
        prereg.preregister(**kwargs)

    kwargs, _manifest = case(safe_root / "second", shared=True)
    _issuer, inventory = prereg.preregister(**kwargs)
    tampered = copy.deepcopy(inventory)
    tampered["outcome_or_trajectory_files_opened_during_attestation"] = False
    tampered.pop("attestation_sha256")
    tampered["attestation_sha256"] = prereg.canonical_sha256(tampered)
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="contract changed"
    ):
        prereg.validate_inventory(tampered)


def test_outputs_are_create_once(safe_root: Path) -> None:
    kwargs, _manifest = case(safe_root, shared=True)
    issuer, inventory = prereg.preregister(**kwargs)
    before = (
        kwargs["issuer_attestation_output"].read_bytes(),
        kwargs["inventory_output"].read_bytes(),
    )
    with pytest.raises(
        prereg.ExecutionInventoryPreregistrationError, match="create-once"
    ):
        prereg.preregister(**kwargs)
    assert before == (
        kwargs["issuer_attestation_output"].read_bytes(),
        kwargs["inventory_output"].read_bytes(),
    )
    assert issuer["attestation_sha256"]
    assert inventory["attestation_sha256"]
