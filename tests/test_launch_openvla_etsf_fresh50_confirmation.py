from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_fresh50_confirmation as launcher  # noqa: E402
import collect_openvla_etsf_event_branches as collector  # noqa: E402
from evaluate_openvla_etsf_counterfactual_sealed import (  # noqa: E402
    validate_evaluation_authorization,
)
from openvla_etsf_counterfactual_oof import make_oof_folds  # noqa: E402
from openvla_etsf_oof_final_contract import (  # noqa: E402
    OOF_PROTOCOL_FORMAT,
    OOF_SELECTION_FORMAT,
    OOF_TEST_POLICY,
    canonical_sha256,
)
from collect_openvla_etsf_event_branches import (  # noqa: E402
    ACTION_DIM,
    BODY,
    CHUNK,
    EVENT_VOCAB,
    HIDDEN_ANCHOR,
    INTERVENTION,
    LANGUAGE_CONTRACT,
    POST_QUERY_ACTION_CONTRACT,
    SCHEMA_VERSION,
    select_seeds,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _fresh_manifest(tmp_path: Path) -> tuple[Path, list[int], list[int]]:
    requested = list(range(10_000, 10_050))
    resolved = list(range(20_000, 20_050))
    candidate = tmp_path / "fresh_candidates.json"
    official = tmp_path / "eval_seeds.json"
    _json(
        candidate,
        {
            "status": "preregistered_unresolved",
            "task": "move_can_pot",
            "candidate_requested_seeds": requested + list(range(30_000, 30_010)),
            "selection_rule": "first_50_unique_resolved_nonofficial_in_frozen_order",
            "freeze_rule": "never_reorder_or_replace_after_policy_access",
        },
    )
    _json(official, {"move_can_pot": {"success_seeds": list(range(150))}})
    rows = [
        {"seed": ask, "requested_seed": ask, "resolved_seed": got}
        for ask, got in zip(requested, resolved)
    ]
    manifest = tmp_path / "fresh_confirmation.json"
    _json(
        manifest,
        {
            "schema_version": 1,
            "status": launcher.FRESH_STATUS,
            "task": "move_can_pot",
            "test": rows,
            "requested_seeds": requested,
            "resolved_seeds": resolved,
            "candidate_manifest": str(candidate.resolve()),
            "candidate_manifest_sha256": launcher.sha256(candidate),
            "official_seed_registry": str(official.resolve()),
            "official_seed_registry_sha256": launcher.sha256(official),
            "selection_rule": "first_50_unique_resolved_nonofficial_in_frozen_order",
            "freeze_rule": "never_reorder_or_replace_after_policy_access",
            "audit": [
                {
                    "requested_seed": ask,
                    "resolved_seed": got,
                    "decision": "selected",
                }
                for ask, got in zip(requested, resolved)
            ],
            "label_access_contract": launcher.FRESH_LABEL_CONTRACT,
        },
    )
    return manifest, requested, resolved


def _ensemble(
    root: Path,
    event_digest: str,
    *,
    guard_enabled: bool = True,
    selected_passes: bool = True,
) -> None:
    root.mkdir(parents=True)
    contract = {
        "event_spec_sha256": event_digest,
        "train_groups": ["move_can_pot|piper|1"],
        "validation_groups": ["move_can_pot|piper|2"],
        "sealed_test_groups": ["move_can_pot|piper|3"],
        "scoring_selection_contract": {
            "selection_data": "validation_only_no_sealed_test"
        },
    }
    scoring = {
        "candidate_id": "success_only",
        "event_weight": 0.0,
        "duration_weight": 0.0,
        "candidate_distance_weight": 0.0,
    }
    scoring_selection = {
        "selection_pool": (
            "pre_guard_evidence_eligible" if selected_passes else "all_grid_candidates"
        ),
        "selected_candidate_id": "success_only",
        "candidates": [
            {
                **scoring,
                "passes_pre_guard_evidence_gate": selected_passes,
                "nonbaseline_proposals": 12,
                "coverage": 0.8,
                "proposal_paired_success_delta_lcb90": 0.0 if selected_passes else -0.2,
            }
        ],
    }
    guard = {
        "enabled": guard_enabled,
        "reason": None if guard_enabled else "no_threshold_met_conservative_validation_gate",
    }
    mirrored = {
        "format": launcher.ENSEMBLE_FORMAT,
        "config": {"structured_events": True},
        "contract": contract,
        "normalization": {"mean": [0.0], "std": [1.0]},
        "duration_scale": 1.0,
        "success_calibration": {"temperature": 1.0},
        "scoring": scoring,
        "scoring_selection": scoring_selection,
        "guard": guard,
        "predicate_contract": {"online_requires_explicit_predicates": True},
        "candidate_contract": {
            "baseline_candidate_name": "deterministic",
            "fallback_index": 0,
        },
    }
    aggregate = root / "counterfactual_ensemble.pt"
    torch.save(mirrored, aggregate)
    members = []
    for seed in launcher.ENSEMBLE_SEEDS:
        member = root / f"counterfactual_seed_{seed}.pt"
        torch.save({"seed": seed}, member)
        members.append(
            {
                "seed": seed,
                "path": str(member.resolve()),
                "sha256": launcher.sha256(member),
            }
        )
    manifest = {
        **mirrored,
        "test_policy": (
            "sealed_identity_attrs_and_sha256_only_label_datasets_not_opened"
        ),
        "ensemble_checkpoint": {
            "path": str(aggregate.resolve()),
            "sha256": launcher.sha256(aggregate),
        },
        "members": members,
    }
    _json(root / "ensemble_manifest.json", manifest)
    _json(root / "launch_audit.json", {"status": "launcher_complete"})


def _oof_ensemble(
    root: Path,
    event_digest: str,
    *,
    guard_enabled: bool = True,
    development_groups: int = 100,
) -> Path:
    root.mkdir(parents=True)
    keys = [
        f"move_can_pot|piper|{1000 + index}"
        for index in range(development_groups)
    ]
    preregistration = make_oof_folds(keys)
    training_per_fold = development_groups - development_groups // 5
    holdout_per_fold = development_groups // 5
    preregistration_sha = preregistration["preregistration_sha256"]
    fold_artifacts = []
    for fold_id in range(5):
        fold_root = root / "folds" / f"fold_{fold_id}"
        fold_root.mkdir(parents=True)
        raw = fold_root / "oof_predictions.pt"
        raw.write_bytes(f"frozen raw fold {fold_id}".encode())
        summary = fold_root / "fold_summary.json"
        _json(
            summary,
            {
                "format": OOF_PROTOCOL_FORMAT,
                "status": "complete",
                "fold_id": fold_id,
                "training_group_count": training_per_fold,
                "oof_holdout_group_count": holdout_per_fold,
                "oof_preregistration_sha256": preregistration_sha,
                "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
                "holdout_labels_first_loaded_after_member_checkpoints": True,
                "raw_predictions": {
                    "path": str(raw.resolve()),
                    "sha256": launcher.sha256(raw),
                },
                "fresh_confirmation_labels_read": False,
            },
        )
        fold_artifacts.append(
            {
                "fold_id": fold_id,
                "summary": str(summary.resolve()),
                "summary_sha256": launcher.sha256(summary),
                "raw_predictions": str(raw.resolve()),
                "raw_predictions_sha256": launcher.sha256(raw),
            }
        )
    authorization = {
        "authorized": True,
        "evidence_tier": "development_oof_authorization_not_confirmation",
        "fresh_confirmation_allowed": True,
        "fresh_confirmation_policy": "one_shot_fresh50_only",
        "total_oof_groups": development_groups,
    }
    scoring = {
        "candidate_id": "success_only",
        "event_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "event_weight": 0.0,
        "duration_weight": 0.0,
        "candidate_distance_weight": 0.0,
        "uncertainty": "success_epistemic_std_plus_mean_model_aleatoric",
    }
    scoring_selection = {
        "selected_candidate_id": "success_only",
        "candidates": [{**scoring, "passes_pre_guard_evidence_gate": True}],
    }
    guard = {"enabled": guard_enabled, "oof_authorization": authorization}
    candidate_authorization = {
        "deployment_candidate_names": [
            "deterministic",
            "sample_blend_0.250",
            "sample_blend_0.500",
            "sample_blend_0.750",
        ],
        "training_only_extra_candidates": ["sample_blend_1.000"],
        "calibration_scoring_guard_use_deployment_candidates_only": True,
    }
    candidate_contract = {
        "baseline_candidate_name": "deterministic",
        "fallback_index": 0,
        **candidate_authorization,
    }
    selection = {
        "format": OOF_SELECTION_FORMAT,
        "status": "complete",
        "oof_prediction_groups": development_groups,
        "oof_preregistration_sha256": preregistration_sha,
        "success_calibration": {"temperature": 1.0},
        "scoring": scoring,
        "scoring_selection": scoring_selection,
        "guard": guard,
        "authorization": authorization,
        "candidate_authorization_contract": candidate_authorization,
        "fold_artifacts": fold_artifacts,
        "fresh_confirmation_labels_read": False,
    }
    selection["selection_sha256"] = canonical_sha256(selection)
    selection_path = root / "oof_selection.json"
    _json(selection_path, selection)
    diagnostics = {
        "format": "etsf_oof_heldout_prediction_diagnostics_v1",
        "status": "complete",
        "evidence_tier": "development_five_fold_heldout_prediction_diagnostics",
        "oof_preregistration_sha256": preregistration_sha,
        "oof_groups": development_groups,
        "fold_count": 5,
        "heldout_groups_per_fold": holdout_per_fold,
        "prediction_source": "each_row_from_unique_owner_fold_model_never_trained_on_that_group",
        "fresh_confirmation_data_or_labels_read": False,
        "authorization_guard_changed": False,
        "diagnostics_are_descriptive_not_an_authorization_or_confirmation_gate": True,
        "success_probability": {},
        "structured_world_model": {"status": "complete"},
        "fold_artifacts": fold_artifacts,
    }
    diagnostics["diagnostics_sha256"] = canonical_sha256(diagnostics)
    diagnostics_path = root / "oof_prediction_diagnostics.json"
    _json(diagnostics_path, diagnostics)
    contract = {
        "trainer": f"five_fold_oof_authorized_refit_all{development_groups}_v1",
        "development_protocol": OOF_PROTOCOL_FORMAT,
        "development_groups": keys,
        "training_groups": keys,
        "train_groups": keys,
        "validation_groups": [],
        "sealed_test_groups": [],
        "sealed_test_files": [],
        "group_files": [],
        "oof_folds": preregistration["folds"],
        "oof_preregistration_sha256": preregistration_sha,
        "training_steps": development_groups,
        "oof_selection": str(selection_path.resolve()),
        "oof_selection_sha256": launcher.sha256(selection_path),
        "oof_prediction_diagnostics": str(diagnostics_path.resolve()),
        "oof_prediction_diagnostics_sha256": launcher.sha256(diagnostics_path),
        "oof_authorization": authorization,
        "candidate_contract": candidate_contract,
        "checkpoint_selection": "fixed_final_step_no_development_metric_selection",
        "sealed_test_access": (
            "fresh50_absent_not_read_one_shot_only_after_oof_authorization"
        ),
        "event_spec_sha256": event_digest,
        "scoring_selection_contract": {
            "selection_data": (
                f"five_fold_oof_all{development_groups}_development_only"
            ),
            "training_groups_per_fold": training_per_fold,
            "holdout_groups_per_fold": holdout_per_fold,
            "oof_preregistration_sha256": preregistration_sha,
            "oof_selection_sha256": launcher.sha256(selection_path),
            "fresh_confirmation_labels_read": False,
        },
        "fresh_confirmation": {
            "authorized": True,
            "required_registry": "explicit_fresh_confirmation",
            "required_groups": 50,
            "access": "not_read_during_development_or_refit",
            "one_shot": True,
        },
    }
    mirrored = {
        "format": launcher.ENSEMBLE_FORMAT,
        "config": {"structured_events": True},
        "contract": contract,
        "normalization": {"mean": [0.0], "std": [1.0]},
        "duration_scale": 1.0,
        "success_calibration": selection["success_calibration"],
        "scoring": scoring,
        "scoring_selection": scoring_selection,
        "guard": guard,
        "predicate_contract": {"online_requires_explicit_predicates": True},
        "candidate_contract": candidate_contract,
    }
    aggregate = root / "counterfactual_ensemble.pt"
    torch.save(mirrored, aggregate)
    members = []
    for seed in launcher.ENSEMBLE_SEEDS:
        member = root / f"counterfactual_seed_{seed}.pt"
        torch.save({"seed": seed}, member)
        members.append(
            {"seed": seed, "path": str(member), "sha256": launcher.sha256(member)}
        )
    manifest = {
        **mirrored,
        "test_policy": OOF_TEST_POLICY,
        "ensemble_checkpoint": {
            "path": str(aggregate),
            "sha256": launcher.sha256(aggregate),
        },
        "members": members,
    }
    manifest_path = root / "ensemble_manifest.json"
    _json(manifest_path, manifest)
    _json(
        root / "training_summary.json",
        {
            "format": OOF_PROTOCOL_FORMAT,
            "status": "complete",
            "development_groups": development_groups,
            "fixed_training_steps": development_groups,
            "oof_authorized": True,
            "fresh_confirmation_labels_read": False,
            "fresh_confirmation_next_action": "one_shot_fresh50_evaluator_only",
            "oof_prediction_diagnostics": str(diagnostics_path.resolve()),
            "oof_prediction_diagnostics_sha256": launcher.sha256(diagnostics_path),
            "member_seeds": list(launcher.ENSEMBLE_SEEDS),
            "ensemble_manifest": str(manifest_path.resolve()),
            "ensemble_manifest_sha256": launcher.sha256(manifest_path),
        },
    )
    return selection_path


def _dry_command(
    tmp_path: Path,
    fresh_manifest: Path,
    counterfactual: Path,
    event_spec: Path,
    python_bin: Path,
) -> list[str]:
    prerequisites = {}
    for name in ("model", "rlinf", "robotwin", "robotwin_code", "data", "factual"):
        path = tmp_path / name
        path.mkdir()
        prerequisites[name] = path
    return [
        sys.executable,
        str(SCRIPTS / "launch_openvla_etsf_fresh50_confirmation.py"),
        "--fresh-seed-manifest", str(fresh_manifest),
        "--counterfactual-root", str(counterfactual),
        "--event-spec", str(event_spec),
        "--data", str(prerequisites["data"]),
        "--factual-root", str(prerequisites["factual"]),
        "--model-path", str(prerequisites["model"]),
        "--rlinf-root", str(prerequisites["rlinf"]),
        "--robotwin-root", str(prerequisites["robotwin"]),
        "--robotwin-code", str(prerequisites["robotwin_code"]),
        "--output", str(tmp_path / "fresh50_output"),
        "--python-bin", str(python_bin),
        "--wait-timeout-seconds", "0",
        "--poll-seconds", "0.01",
        "--dry-run",
    ]


def test_dry_run_freezes_order_and_preserves_venv_symlink(tmp_path: Path) -> None:
    assert launcher.DEFAULT_COUNTERFACTUAL_OUTPUT == Path(
        "/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry1_20260827"
    )
    assert launcher.DEFAULT_FRESH_MANIFEST == Path(
        "/home/user/etsf_event_world_model_code_20260827/artifacts/protocol/"
        "fresh_confirmation_seeds_20260827.json"
    )
    fresh_manifest, requested, resolved = _fresh_manifest(tmp_path)
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {"chains": {}, "calibration": {}})
    counterfactual = tmp_path / "counterfactual"
    _ensemble(counterfactual, launcher.sha256(event_spec))
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable).resolve())

    completed = subprocess.run(
        _dry_command(
            tmp_path, fresh_manifest, counterfactual, event_spec, venv_python
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(
        completed.stdout.removeprefix("FRESH50_CONFIRMATION_DRY_RUN=")
    )
    assert audit["fresh"]["requested_seeds"] == requested
    assert audit["fresh"]["resolved_seeds"] == resolved
    assert audit["contract"]["stage_order"] == [
        "collector",
        "evaluator_once",
        "progress_after_evaluation",
    ]
    expected_python = str(venv_python.absolute())
    assert all(
        row["argv"][0] == expected_python for row in audit["commands"].values()
    )
    assert expected_python != str(venv_python.resolve())
    collector = audit["commands"]["collector"]["argv"]
    assert collector[collector.index("--seeds-file") + 1] == str(
        fresh_manifest.resolve()
    )
    assert "--allow-unregistered-seeds" in collector
    assert collector[collector.index("--seeds-key") + 1] == "test"
    assert "--fresh-seed-manifest" in collector
    assert not (tmp_path / "fresh50_output").exists()
    assert audit["sealed_labels_read_by_launcher_before_evaluation"] is False


@pytest.mark.parametrize(
    ("guard_enabled", "selected_passes", "match"),
    [
        (False, True, "validation guard is disabled"),
        (True, False, "pre_guard_evidence_gate"),
    ],
)
def test_fresh_consumption_rejects_failed_validation_evidence(
    tmp_path: Path,
    guard_enabled: bool,
    selected_passes: bool,
    match: str,
) -> None:
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    root = tmp_path / "counterfactual"
    _ensemble(
        root,
        launcher.sha256(event_spec),
        guard_enabled=guard_enabled,
        selected_passes=selected_passes,
    )
    with pytest.raises(RuntimeError, match=match):
        launcher.audit_frozen_ensemble(root, launcher.sha256(event_spec))


def test_authorized_oof_final_is_accepted_only_for_one_shot_fresh(
    tmp_path: Path,
) -> None:
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    root = tmp_path / "oof_final"
    _oof_ensemble(
        root, launcher.sha256(event_spec), development_groups=250
    )
    audit = launcher.audit_frozen_ensemble(root, launcher.sha256(event_spec))
    assert audit["authorization_mode"] == "authorized_oof_final"
    assert audit["oof_authorization"]["guard_enabled"] is True
    assert len(audit["oof_authorization"]["fold_artifacts"]) == 5
    manifest_path = root / "ensemble_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = tmp_path / "fresh_sealed"
    sealed.mkdir()
    with pytest.raises(RuntimeError, match="only consume one-shot fresh50"):
        validate_evaluation_authorization(
            manifest_path,
            manifest,
            fresh_manifest_present=False,
            sealed_root=sealed,
        )
    authorized = validate_evaluation_authorization(
        manifest_path,
        manifest,
        fresh_manifest_present=True,
        sealed_root=sealed,
    )
    assert authorized["mode"] == "authorized_oof_final"


def test_oof_selection_sha_tamper_and_disabled_guard_fail_closed(
    tmp_path: Path,
) -> None:
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    root = tmp_path / "tampered"
    selection = _oof_ensemble(root, launcher.sha256(event_spec))
    selection.write_text(selection.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection file SHA256"):
        launcher.audit_frozen_ensemble(root, launcher.sha256(event_spec))

    disabled = tmp_path / "disabled"
    _oof_ensemble(disabled, launcher.sha256(event_spec), guard_enabled=False)
    with pytest.raises(RuntimeError, match="guard is disabled|guard is disabled or not mirrored"):
        launcher.audit_frozen_ensemble(disabled, launcher.sha256(event_spec))


def test_official_overlap_and_seeds_file_contract_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_manifest, requested, _ = _fresh_manifest(tmp_path)
    value = json.loads(fresh_manifest.read_text(encoding="utf-8"))
    official = Path(value["official_seed_registry"])
    _json(official, {"move_can_pot": {"success_seeds": [requested[0]]}})
    value["official_seed_registry_sha256"] = launcher.sha256(official)
    _json(fresh_manifest, value)
    with pytest.raises(RuntimeError, match="overlap official"):
        launcher.audit_fresh_manifest(fresh_manifest, "move_can_pot")

    explicit = tmp_path / "explicit_seeds.json"
    _json(explicit, {"test": [{"seed": 10_000}, {"seed": 10_001}]})
    registry = tmp_path / "official.json"
    _json(registry, {"move_can_pot": {"success_seeds": [1, 2, 3]}})
    args = argparse.Namespace(
        seeds_file=explicit,
        seeds_key="test",
        seeds=None,
        task="move_can_pot",
        limit=150,
        offset=0,
        allow_unregistered_seeds=True,
    )
    monkeypatch.setattr(
        collector,
        "load_official_seeds",
        lambda path, task, limit, offset: [1, 2, 3],
        raising=False,
    )
    assert select_seeds(args, registry) == [10_000, 10_001]


def test_collection_identity_reads_only_identity_attrs_not_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "collection"
    groups = root / "groups"
    groups.mkdir(parents=True)
    group = groups / "group_000.hdf5"
    requested, resolved = 10_000, 20_000
    candidate_names = [
        "deterministic",
        "sample_blend_0.25",
        "sample_blend_0.5",
        "sample_blend_0.75",
    ]
    with h5py.File(group, "w") as handle:
        handle.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": requested,
                "requested_seed": requested,
                "resolved_seed": resolved,
                "candidate_count": 4,
                "language_contract": LANGUAGE_CONTRACT,
                "branch_instruction_consistent": True,
                "intervention": INTERVENTION,
                "post_query_action_contract": POST_QUERY_ACTION_CONTRACT,
            }
        )
        handle.create_dataset("candidate_names", data=[name.encode() for name in candidate_names])
        # Deliberately malformed label-shaped data: the launcher must not open it.
        handle.create_dataset("success", data=[1, 0])
    model = tmp_path / "model"
    model.mkdir()
    event_spec = tmp_path / "event_spec.json"
    _json(event_spec, {})
    fresh = {
        "task": "move_can_pot",
        "sha256": "a" * 64,
        "requested_seeds": [requested],
        "rows": [
            {
                "seed": requested,
                "requested_seed": requested,
                "resolved_seed": resolved,
            }
        ],
    }
    identity = {
        "format": launcher.IDENTITY_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "collecting",
        "task": "move_can_pot",
        "body": BODY,
        "model_path": str(model),
        "requested_seeds": [requested],
        "resolved_seeds": [resolved],
        "seed_registry": "explicit_fresh_confirmation",
        "fresh_seed_manifest_sha256": fresh["sha256"],
        "candidate_count": 4,
        "blends": [0.25, 0.5, 0.75],
        "temperature": 0.7,
        "top_k": 4,
        "preserve_grippers": True,
        "intervention": INTERVENTION,
        "language_contract": LANGUAGE_CONTRACT,
        "event_vocab": list(EVENT_VOCAB),
        "event_spec_sha256": launcher.sha256(event_spec),
        "hidden_dim": 4096,
        "hidden_anchor": HIDDEN_ANCHOR,
        "action_dim": ACTION_DIM,
        "action_chunk": CHUNK,
        "label_access_contract": "identity_only_no_success_steps_event_or_outcome_fields",
        "hdf5_sha256_pre_evaluation": "not_computed",
        "groups": [
            {
                "index": 0,
                "seed": requested,
                "requested_seed": requested,
                "resolved_seed": resolved,
                "path": group.name,
                "candidate_names": candidate_names,
                "status": "collected",
            }
        ],
    }
    _json(root / "collection_identity.json", identity)
    audit = launcher.validate_collection_identity(
        root,
        fresh,
        launcher.sha256(event_spec),
        model,
        require_complete=False,
    )
    assert audit["labels_read"] is False
    assert audit["hdf5_sha256_computed"] is False
