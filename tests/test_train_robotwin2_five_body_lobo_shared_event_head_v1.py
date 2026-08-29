from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_multibody_canonical_event_world_model as core  # noqa: E402
import preregister_robotwin2_move_can_pot_five_body_lobo_v1 as prereg  # noqa: E402
import verify_robotwin2_move_can_pot_public_materialization_v1 as verifier  # noqa: E402
from train_robotwin2_five_body_lobo_shared_event_head_v1 import (  # noqa: E402
    ACTOR_FORMAT,
    BINDING_FORMAT,
    BODIES,
    CANONICAL_ACTION_SCHEMA,
    CANONICAL_STATE_SCHEMA,
    DATASET_REPO,
    DATASET_REVISION,
    MANIFEST_FORMAT,
    MATERIALIZATION_FORMAT,
    PREREGISTRATION_SHA256,
    TASK,
    FiveBodyContractError,
    build_preflight_receipt,
    canonical_sha256,
    load_binding,
    materialize_source_rows,
    sha256_file,
    sha256_tree,
    source_group_split,
)


def _signed(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["logical_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


def _group(path: Path, offset: float) -> None:
    count, horizon = 4, 3
    np.savez(
        path,
        state=np.full((count, core.STATE_DIM), offset, dtype=np.float32),
        actions=np.full((count, horizon, core.ACTION_DIM), offset, dtype=np.float32),
        action_mask=np.ones((count, horizon), dtype=np.float32),
        current_event_id=np.zeros(count, dtype=np.int64),
        post_event_id=np.asarray([1, 2, 3, 4], dtype=np.int64),
        post_event_mask=np.ones(count, dtype=np.float32),
        next_event_id=np.asarray([1, 2, 3, 4], dtype=np.int64),
        next_event_mask=np.ones(count, dtype=np.float32),
        duration=np.asarray([1, 2, 3, 4], dtype=np.float32),
        duration_observed=np.ones(count, dtype=np.float32),
        duration_mask=np.ones(count, dtype=np.float32),
        success=np.asarray([0, 1, 0, 1], dtype=np.float32),
        success_mask=np.ones(count, dtype=np.float32),
        recovery=np.asarray([0, 0, 1, 0], dtype=np.float32),
        recovery_mask=np.ones(count, dtype=np.float32),
        object_delta=np.full((count, core.OBJECT_DELTA_DIM), 0.1, dtype=np.float32),
        object_delta_mask=np.ones(count, dtype=np.float32),
        candidate_index=np.arange(count, dtype=np.int64),
        dt=np.full(count, 0.1, dtype=np.float32),
    )


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    files = [
        {
            "path": f"dataset/move_can_pot/file-{index}.zip",
            "size_bytes": index + 1,
            "expected_payload_sha256": f"{index + 1:064x}",
            "observed_payload_sha256": f"{index + 1:064x}",
            "size_match": True,
            "payload_sha256_match": True,
            "zip_central_directory_audit": {
                "central_directory_read_only": True,
                "member_payload_bytes_read": 0,
            },
        }
        for index in range(11)
    ]
    materialization_unsigned = {
        "format": MATERIALIZATION_FORMAT,
        "status": verifier.STATUS,
        "materialized": True,
        "no_missing_or_extra_official_task_files": True,
        "all_exact_sizes_verified": True,
        "all_exact_archive_payload_sha256_verified": True,
        "hf_repo_id": DATASET_REPO,
        "hf_repo_revision": DATASET_REVISION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "official_file_count": 11,
        "official_total_size_bytes": sum(range(1, 12)),
        "files": files,
        "read_boundary": {
            "zip_member_payload_bytes_read": 0,
            "archive_extracted": False,
            "pickle_payload_opened_or_deserialized": False,
            "numpy_payload_opened_or_deserialized": False,
            "torch_payload_opened_or_deserialized": False,
        },
        "implementation_binding": {
            "verifier_module": Path(verifier.__file__).name,
            "verifier_file_sha256": verifier._file_sha256(Path(verifier.__file__).resolve()),
            "preregistration_module": Path(prereg.__file__).name,
            "preregistration_module_file_sha256": verifier._file_sha256(
                Path(prereg.__file__).resolve()
            ),
        },
        "authority": {
            "download_completeness_attested": True,
            "training_authorized": False,
            "evaluation_authorized": False,
            "simulator_execution_authorized": False,
            "checkpoint_selection_or_promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_performance_claim_authorized": False,
        },
    }
    materialization = {
        **materialization_unsigned,
        "materialization_receipt_sha256": verifier.canonical_sha256(
            materialization_unsigned
        ),
    }
    materialization_path = tmp_path / "materialization.json"
    materialization_sha = _write_json(materialization_path, materialization)

    actors = {}
    for index, body in enumerate(BODIES):
        checkpoint_path = tmp_path / f"{body}-actor.ckpt"
        if body == "piper":
            checkpoint_path.mkdir()
            (checkpoint_path / "config.json").write_text("{}\n", encoding="utf-8")
            (checkpoint_path / "model.safetensors").write_bytes(b"frozen-piper")
            checkpoint_kind = "directory_tree"
            checkpoint_sha256 = sha256_tree(checkpoint_path)[0]
        else:
            checkpoint_path.write_bytes(f"frozen-{body}".encode())
            checkpoint_kind = "file"
            checkpoint_sha256 = sha256_file(checkpoint_path)
        actors[body] = {
            "family": "synthetic-test-native-actor",
            "frozen": True,
            "optimizer_updates_allowed": False,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_sha256": checkpoint_sha256,
            "sampling_contract_sha256": f"{index + 40:064x}",
            "candidate_count": 4,
            "candidate_zero_is_actor_baseline": True,
            "same_ordered_candidate_set_for_baseline_and_etsf": True,
        }
    actor_authority = _signed(
        {
            "format": ACTOR_FORMAT,
            "task": TASK,
            "actors": actors,
        }
    )
    actor_path = tmp_path / "actors.json"
    actor_sha = _write_json(actor_path, actor_authority)

    body_bindings = {}
    for body_index, body in enumerate(BODIES):
        groups = []
        for condition in ("clean", "randomized"):
            for group_index in range(2):
                name = f"{body}-{condition}-{group_index}.npz"
                group_path = tmp_path / name
                _group(group_path, float(body_index + group_index + 1))
                groups.append(
                    {
                        "group_id": name.removesuffix(".npz"),
                        "condition": condition,
                        "requested_seed": body_index * 100 + group_index,
                        "path": name,
                        "sha256": sha256_file(group_path),
                    }
                )
        manifest = _signed(
            {
                "format": MANIFEST_FORMAT,
                "dataset_repo": DATASET_REPO,
                "dataset_revision": DATASET_REVISION,
                "task": TASK,
                "body": body,
                "schema_adapter": {
                    "kind": "analytic_label_free_canonical_v1",
                    "trainable": False,
                    "labels_or_outcomes_used_to_fit": False,
                    "heldout_supervision_allowed": False,
                    "state_dim": core.STATE_DIM,
                    "action_dim": core.ACTION_DIM,
                    "state_schema": CANONICAL_STATE_SCHEMA,
                    "action_schema": CANONICAL_ACTION_SCHEMA,
                    "elapsed_time_unit": "seconds",
                    "duration_unit": "seconds",
                    "event_names": list(core.CANONICAL_EVENTS),
                    "implementation_sha256": f"{body_index + 60:064x}",
                },
                "groups": groups,
            }
        )
        manifest_path = tmp_path / f"{body}-manifest.json"
        body_bindings[body] = {
            "path": manifest_path.name,
            "sha256": _write_json(manifest_path, manifest),
        }

    binding = _signed(
        {
            "format": BINDING_FORMAT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "task": TASK,
            "heldout_labels_may_train_fit_calibrate_or_select": False,
            "canonical_shared_body_rows": 1,
            "execution_authority": {
                "explicit_user_training_request_recorded": True,
                "public_data_only": True,
                "protected_internal_data_allowed": False,
                "remote_cuda_only": True,
            },
            "materialization_receipt": {
                "path": materialization_path.name,
                "sha256": materialization_sha,
            },
            "actor_authority": {"path": actor_path.name, "sha256": actor_sha},
            "body_manifests": body_bindings,
        }
    )
    binding_path = tmp_path / "binding.json"
    return binding_path, _write_json(binding_path, binding)


def test_preflight_is_five_fold_source_only_and_payload_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("preflight opened a transition NPZ")
    ))
    for heldout in BODIES:
        receipt = build_preflight_receipt(audit, held_out_body=heldout, split_seed=17)
        assert heldout not in receipt["source_bodies"]
        assert len(receipt["source_bodies"]) == 4
        assert receipt["heldout_group_npz_opened"] == 0
        assert receipt["heldout_specific_trainable_parameters"] == 0
        assert receipt["model_body_rows"] == 1
        assert receipt["actor_frozen"] is True


def test_heldout_payload_is_not_stat_hashed_or_deserialized_in_preflight(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_path = tmp_path / decoded["body_manifests"]["franka"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for group in manifest["groups"]:
        (tmp_path / group["path"]).unlink()
    # A strict Franka-heldout preflight consumes only the declared paths and
    # commitments.  Missing target payloads therefore cannot be observed.
    audit = load_binding(binding, digest)
    receipt = build_preflight_receipt(
        audit, held_out_body="franka", split_seed=23
    )
    assert receipt["heldout_group_payload_bytes_read"] == 0
    assert receipt["heldout_group_payload_deserialized"] == 0

    # The same missing files fail once Franka becomes a source body and its
    # source payload boundary is deliberately crossed.
    train, _, _ = source_group_split(audit, held_out_body="piper", split_seed=23)
    franka = [group for group in train if group["body"] == "franka"]
    with pytest.raises(FiveBodyContractError, match="missing/tampered"):
        materialize_source_rows(franka, held_out_body="piper")


def test_source_materialization_never_accepts_heldout_and_model_has_one_body_row(
    tmp_path: Path,
) -> None:
    binding, digest = _fixture(tmp_path)
    audit = load_binding(binding, digest)
    train, validation, heldout = source_group_split(
        audit, held_out_body="franka", split_seed=19
    )
    assert all(row["body"] != "franka" for row in train + validation)
    assert all("body" not in row for row in heldout)
    rows = materialize_source_rows(train, held_out_body="franka")
    assert rows and all(row["body"] != "franka" for row in rows)
    mapping = {body: 0 for body in BODIES if body != "franka"}
    dataset = core.TransitionDataset(rows, mapping)
    batch = core.collate_rows([dataset[0], dataset[1]])
    model = core.MultibodyCanonicalEventWorldModel(
        core.ModelConfig(body_count=1, action_schema_count=1, dropout=0.0)
    ).eval()
    assert model.clock.body_beta.weight.shape[0] == 1
    assert model.action.schema_count == 1
    assert set(batch["body_id"].tolist()) == {0}
    with torch.no_grad():
        output = model(batch)
    assert output["success_logit"].shape == (2,)
    with pytest.raises(FiveBodyContractError, match="held-out group"):
        materialize_source_rows(
            [{**audit["manifests"]["franka"]["groups"][0], "body": "franka"}],
            held_out_body="franka",
        )


def test_tampered_or_supervised_adapter_fails_closed(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    manifest_binding = decoded["body_manifests"]["piper"]
    manifest_path = tmp_path / manifest_binding["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_adapter"]["labels_or_outcomes_used_to_fit"] = True
    unsigned = dict(manifest)
    unsigned.pop("logical_sha256")
    manifest["logical_sha256"] = canonical_sha256(unsigned)
    manifest_binding["sha256"] = _write_json(manifest_path, manifest)
    binding_unsigned = dict(decoded)
    binding_unsigned.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(binding_unsigned)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="analytic/label-free"):
        load_binding(binding, digest)


def test_incomplete_public_download_receipt_fails_closed(tmp_path: Path) -> None:
    binding, _ = _fixture(tmp_path)
    decoded = json.loads(binding.read_text(encoding="utf-8"))
    receipt_path = tmp_path / decoded["materialization_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "download_incomplete"
    unsigned = dict(receipt)
    unsigned.pop("materialization_receipt_sha256")
    receipt["materialization_receipt_sha256"] = verifier.canonical_sha256(unsigned)
    decoded["materialization_receipt"]["sha256"] = _write_json(receipt_path, receipt)
    unsigned_binding = dict(decoded)
    unsigned_binding.pop("logical_sha256")
    decoded["logical_sha256"] = canonical_sha256(unsigned_binding)
    digest = _write_json(binding, decoded)
    with pytest.raises(FiveBodyContractError, match="not verified"):
        load_binding(binding, digest)
