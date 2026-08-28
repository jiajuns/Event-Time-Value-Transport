#!/usr/bin/env python3
"""Read-only validation for an OOF-authorized final ETSF ensemble.

This module is deliberately independent from the OOF trainer/launcher.  Both
fresh orchestration and the one-shot evaluator use it before any fresh outcome
manifest or label dataset is opened.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ENSEMBLE_FORMAT = "etsf_counterfactual_ensemble_v1"
OOF_PROTOCOL_FORMAT = "etsf_counterfactual_five_fold_oof_v1"
OOF_SELECTION_FORMAT = "etsf_counterfactual_oof_selection_v1"
OOF_PREDICTION_DIAGNOSTICS_FORMAT = "etsf_oof_heldout_prediction_diagnostics_v1"
OOF_TEST_POLICY = "fresh50_one_shot_only_after_oof_authorization"
SUPPORTED_EXPECTED_GROUPS = (100, 250)
EXPECTED_FOLDS = 5
EXPECTED_MEMBER_SEEDS = (20260827, 20260828, 20260829)
DEPLOYMENT_CANDIDATE_NAMES = [
    "deterministic",
    "sample_blend_0.250",
    "sample_blend_0.500",
    "sample_blend_0.750",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_equivalent(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":"), default=str) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), default=str
    )


def resolve_artifact(recorded: str, anchor: Path) -> Path:
    path = Path(recorded).expanduser()
    if path.is_file():
        return path.resolve()
    portable = anchor / path.name
    if portable.is_file():
        return portable.resolve()
    raise FileNotFoundError(path)


def _require_outside(path: Path, forbidden_root: Path | None) -> None:
    if forbidden_root is None:
        return
    root = forbidden_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if resolved == root or root in resolved.parents:
        raise RuntimeError("OOF development artifact aliases the sealed data root")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"OOF final contract lacks {name}")
    return value


def _validate_embedded_folds(contract: Mapping[str, Any]) -> tuple[int, int, int]:
    development = contract.get("development_groups")
    training = contract.get("training_groups")
    folds = contract.get("oof_folds")
    expected_groups = len(development) if isinstance(development, list) else -1
    if (
        expected_groups not in SUPPORTED_EXPECTED_GROUPS
        or len(set(map(str, development or []))) != expected_groups
        or list(map(str, training or [])) != list(map(str, development))
        or not isinstance(folds, list)
        or len(folds) != EXPECTED_FOLDS
    ):
        raise RuntimeError("OOF final development/fold manifest is incomplete")
    holdout_groups = expected_groups // EXPECTED_FOLDS
    training_groups = expected_groups - holdout_groups
    keys = set(map(str, development))
    holdout_owner: dict[str, int] = {}
    for fold_id, fold in enumerate(folds):
        if not isinstance(fold, Mapping) or int(fold.get("fold_id", -1)) != fold_id:
            raise RuntimeError("OOF final fold id/order changed")
        train = set(map(str, fold.get("training_groups", [])))
        holdout = set(map(str, fold.get("oof_holdout_groups", [])))
        if (
            len(train) != training_groups
            or len(holdout) != holdout_groups
            or train & holdout
            or train | holdout != keys
            or fold.get("checkpoint_selection")
            != "fixed_final_step_no_holdout_early_stop"
        ):
            raise RuntimeError("OOF final embedded fold contract changed")
        for key in holdout:
            if key in holdout_owner:
                raise RuntimeError("OOF final holdout groups are not unique")
            holdout_owner[key] = fold_id
    if set(holdout_owner) != keys:
        raise RuntimeError("OOF final folds do not cover development groups exactly once")
    return expected_groups, training_groups, holdout_groups


def _validate_fold_artifacts(
    selection: Mapping[str, Any],
    selection_path: Path,
    preregistration_sha256: str,
    forbidden_root: Path | None,
    expected_training_groups: int,
    expected_holdout_groups: int,
) -> list[dict[str, Any]]:
    artifacts = selection.get("fold_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_FOLDS:
        raise RuntimeError("OOF selection lacks five frozen fold artifacts")
    audited = []
    for fold_id, record in enumerate(artifacts):
        if not isinstance(record, Mapping) or int(record.get("fold_id", -1)) != fold_id:
            raise RuntimeError("OOF fold artifact id/order changed")
        summary_path = resolve_artifact(str(record.get("summary", "")), selection_path.parent)
        raw_path = resolve_artifact(
            str(record.get("raw_predictions", "")), selection_path.parent
        )
        _require_outside(summary_path, forbidden_root)
        _require_outside(raw_path, forbidden_root)
        summary_digest = sha256(summary_path)
        raw_digest = sha256(raw_path)
        if summary_digest != str(record.get("summary_sha256", "")):
            raise RuntimeError("OOF fold summary artifact SHA256 mismatch")
        if raw_digest != str(record.get("raw_predictions_sha256", "")):
            raise RuntimeError("OOF raw prediction artifact SHA256 mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        raw_record = summary.get("raw_predictions") if isinstance(summary, Mapping) else None
        if (
            not isinstance(summary, Mapping)
            or summary.get("format") != OOF_PROTOCOL_FORMAT
            or summary.get("status") != "complete"
            or int(summary.get("fold_id", -1)) != fold_id
            or summary.get("oof_preregistration_sha256") != preregistration_sha256
            or summary.get("checkpoint_selection")
            != "fixed_final_step_no_holdout_early_stop"
            or summary.get("holdout_labels_first_loaded_after_member_checkpoints")
            is not True
            or summary.get("fresh_confirmation_labels_read") is not False
            or int(summary.get("training_group_count", -1))
            != expected_training_groups
            or int(summary.get("oof_holdout_group_count", -1))
            != expected_holdout_groups
            or not isinstance(raw_record, Mapping)
            or str(raw_record.get("sha256", "")) != raw_digest
        ):
            raise RuntimeError("OOF fold summary contract mismatch")
        recorded_raw = resolve_artifact(
            str(raw_record.get("path", "")), summary_path.parent
        )
        if recorded_raw != raw_path:
            raise RuntimeError("OOF fold summary points to another raw artifact")
        audited.append(
            {
                "fold_id": fold_id,
                "summary": str(summary_path),
                "summary_sha256": summary_digest,
                "raw_predictions": str(raw_path),
                "raw_predictions_sha256": raw_digest,
            }
        )
    return audited


def validate_authorized_oof_final(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    forbidden_root: Path | None = None,
) -> dict[str, Any]:
    """Validate all available OOF authorization provenance and artifact SHA."""

    manifest_path = manifest_path.expanduser().resolve()
    if (
        manifest.get("format") != ENSEMBLE_FORMAT
        or manifest.get("test_policy") != OOF_TEST_POLICY
    ):
        raise RuntimeError("ensemble is not an OOF-authorized fresh-only final")
    contract = _mapping(manifest.get("contract"), "contract")
    if (
        contract.get("development_protocol") != OOF_PROTOCOL_FORMAT
        or contract.get("checkpoint_selection")
        != "fixed_final_step_no_development_metric_selection"
        or contract.get("sealed_test_access")
        != "fresh50_absent_not_read_one_shot_only_after_oof_authorization"
        or contract.get("validation_groups") != []
        or contract.get("sealed_test_groups") != []
        or contract.get("sealed_test_files") != []
    ):
        raise RuntimeError("OOF final refit/training contract changed")
    expected_groups, expected_training_groups, expected_holdout_groups = (
        _validate_embedded_folds(contract)
    )
    if contract.get("trainer") != (
        f"five_fold_oof_authorized_refit_all{expected_groups}_v1"
    ):
        raise RuntimeError("OOF final trainer does not match development cardinality")
    if int(contract.get("training_steps", -1)) != expected_groups:
        raise RuntimeError("OOF final training exposure budget changed")
    preregistration_sha256 = str(contract.get("oof_preregistration_sha256", ""))
    if len(preregistration_sha256) != 64:
        raise RuntimeError("OOF final lacks preregistration SHA256")

    fresh = _mapping(contract.get("fresh_confirmation"), "fresh_confirmation")
    if (
        fresh.get("authorized") is not True
        or fresh.get("required_registry") != "explicit_fresh_confirmation"
        or int(fresh.get("required_groups", -1)) != 50
        or fresh.get("access") != "not_read_during_development_or_refit"
        or fresh.get("one_shot") is not True
    ):
        raise RuntimeError("OOF final does not authorize one-shot fresh50")
    candidate_contract = _mapping(
        contract.get("candidate_contract"), "candidate_contract"
    )
    if (
        candidate_contract.get("baseline_candidate_name") != "deterministic"
        or int(candidate_contract.get("fallback_index", -1)) != 0
        or candidate_contract.get("deployment_candidate_names")
        != DEPLOYMENT_CANDIDATE_NAMES
        or candidate_contract.get("training_only_extra_candidates")
        != ["sample_blend_1.000"]
        or candidate_contract.get(
            "calibration_scoring_guard_use_deployment_candidates_only"
        )
        is not True
        or not json_equivalent(manifest.get("candidate_contract"), candidate_contract)
    ):
        raise RuntimeError("OOF final candidate schedule is not fresh50-compatible")

    selection_path = resolve_artifact(
        str(contract.get("oof_selection", "")), manifest_path.parent
    )
    _require_outside(selection_path, forbidden_root)
    selection_digest = sha256(selection_path)
    if selection_digest != str(contract.get("oof_selection_sha256", "")):
        raise RuntimeError("OOF selection file SHA256 mismatch")
    diagnostics_path = resolve_artifact(
        str(contract.get("oof_prediction_diagnostics", "")), manifest_path.parent
    )
    _require_outside(diagnostics_path, forbidden_root)
    diagnostics_digest = sha256(diagnostics_path)
    if diagnostics_digest != str(
        contract.get("oof_prediction_diagnostics_sha256", "")
    ):
        raise RuntimeError("OOF prediction diagnostics file SHA256 mismatch")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("OOF prediction diagnostics must contain a JSON object")
    unsigned_diagnostics = dict(diagnostics)
    internal_diagnostics_digest = str(
        unsigned_diagnostics.pop("diagnostics_sha256", "")
    )
    structured_diagnostics = diagnostics.get("structured_world_model")
    if (
        diagnostics.get("format") != OOF_PREDICTION_DIAGNOSTICS_FORMAT
        or diagnostics.get("status") != "complete"
        or internal_diagnostics_digest != canonical_sha256(unsigned_diagnostics)
        or diagnostics.get("oof_preregistration_sha256")
        != preregistration_sha256
        or int(diagnostics.get("oof_groups", -1)) != expected_groups
        or diagnostics.get("fresh_confirmation_data_or_labels_read") is not False
        or diagnostics.get("authorization_guard_changed") is not False
        or not isinstance(structured_diagnostics, Mapping)
        or structured_diagnostics.get("status") != "complete"
    ):
        raise RuntimeError("OOF structured prediction diagnostics changed or incomplete")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise RuntimeError("OOF selection must contain a JSON object")
    unsigned = dict(selection)
    internal_digest = str(unsigned.pop("selection_sha256", ""))
    if (
        selection.get("format") != OOF_SELECTION_FORMAT
        or selection.get("status") != "complete"
        or internal_digest != canonical_sha256(unsigned)
        or selection.get("oof_preregistration_sha256") != preregistration_sha256
        or selection.get("fresh_confirmation_labels_read") is not False
        or int(selection.get("oof_prediction_groups", -1)) != expected_groups
    ):
        raise RuntimeError("OOF selection manifest/signature mismatch")
    selection_contract = _mapping(
        contract.get("scoring_selection_contract"), "scoring_selection_contract"
    )
    if (
        selection_contract.get("selection_data")
        != f"five_fold_oof_all{expected_groups}_development_only"
        or int(selection_contract.get("training_groups_per_fold", -1))
        != expected_training_groups
        or int(selection_contract.get("holdout_groups_per_fold", -1))
        != expected_holdout_groups
        or selection_contract.get("oof_preregistration_sha256")
        != preregistration_sha256
        or selection_contract.get("oof_selection_sha256") != selection_digest
        or selection_contract.get("fresh_confirmation_labels_read") is not False
    ):
        raise RuntimeError("OOF final scoring-selection provenance changed")
    if not json_equivalent(
        selection.get("candidate_authorization_contract"),
        {
            key: candidate_contract[key]
            for key in (
                "deployment_candidate_names",
                "training_only_extra_candidates",
                "calibration_scoring_guard_use_deployment_candidates_only",
            )
        },
    ):
        raise RuntimeError("OOF candidate authorization contract changed")

    authorization = _mapping(selection.get("authorization"), "authorization")
    if (
        authorization.get("authorized") is not True
        or authorization.get("fresh_confirmation_allowed") is not True
        or authorization.get("fresh_confirmation_policy") != "one_shot_fresh50_only"
        or authorization.get("evidence_tier")
        != "development_oof_authorization_not_confirmation"
        or int(authorization.get("total_oof_groups", -1)) != expected_groups
        or not json_equivalent(contract.get("oof_authorization"), authorization)
    ):
        raise RuntimeError("OOF selection authorization is absent or changed")
    guard = _mapping(selection.get("guard"), "guard")
    if (
        guard.get("enabled") is not True
        or not json_equivalent(guard.get("oof_authorization"), authorization)
        or not json_equivalent(manifest.get("guard"), guard)
    ):
        raise RuntimeError("OOF frozen guard is disabled or not mirrored")
    for key in ("scoring", "scoring_selection", "success_calibration"):
        if not json_equivalent(manifest.get(key), selection.get(key)):
            raise RuntimeError(f"OOF final {key} differs from frozen selection")

    fold_artifacts = _validate_fold_artifacts(
        selection,
        selection_path,
        preregistration_sha256,
        forbidden_root,
        expected_training_groups,
        expected_holdout_groups,
    )
    summary_path = manifest_path.parent / "training_summary.json"
    _require_outside(summary_path, forbidden_root)
    if not summary_path.is_file():
        raise RuntimeError("OOF final training_summary.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_manifest = resolve_artifact(
        str(summary.get("ensemble_manifest", "")), summary_path.parent
    )
    if (
        summary.get("format") != OOF_PROTOCOL_FORMAT
        or summary.get("status") != "complete"
        or summary.get("oof_authorized") is not True
        or summary.get("fresh_confirmation_labels_read") is not False
        or summary.get("fresh_confirmation_next_action")
        != "one_shot_fresh50_evaluator_only"
        or list(summary.get("member_seeds", [])) != list(EXPECTED_MEMBER_SEEDS)
        or int(summary.get("development_groups", -1)) != expected_groups
        or int(summary.get("fixed_training_steps", -1)) != expected_groups
        or summary_manifest != manifest_path
        or str(summary.get("ensemble_manifest_sha256", "")) != sha256(manifest_path)
        or resolve_artifact(
            str(summary.get("oof_prediction_diagnostics", "")), summary_path.parent
        )
        != diagnostics_path
        or str(summary.get("oof_prediction_diagnostics_sha256", ""))
        != diagnostics_digest
    ):
        raise RuntimeError("OOF final summary/manifest SHA contract mismatch")

    # Newer launchers may add the preregistration artifact itself.  When
    # present it is mandatory and verified, while current v1 final outputs
    # remain compatible through the canonical preregistration digest above.
    optional_manifest = contract.get("oof_manifest")
    if optional_manifest is not None:
        oof_manifest_path = resolve_artifact(str(optional_manifest), manifest_path.parent)
        _require_outside(oof_manifest_path, forbidden_root)
        if sha256(oof_manifest_path) != str(contract.get("oof_manifest_sha256", "")):
            raise RuntimeError("OOF preregistration manifest file SHA256 mismatch")
        oof_manifest = json.loads(oof_manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(oof_manifest, Mapping)
            or oof_manifest.get("format") != OOF_PROTOCOL_FORMAT
            or oof_manifest.get("preregistration_sha256") != preregistration_sha256
            or int(oof_manifest.get("expected_groups", -1)) != expected_groups
            or int(oof_manifest.get("training_steps", -1)) != expected_groups
        ):
            raise RuntimeError("OOF preregistration manifest contract mismatch")

    return {
        "mode": "authorized_oof_final",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "training_summary": str(summary_path.resolve()),
        "training_summary_sha256": sha256(summary_path),
        "selection": str(selection_path),
        "selection_file_sha256": selection_digest,
        "selection_internal_sha256": internal_digest,
        "prediction_diagnostics": str(diagnostics_path),
        "prediction_diagnostics_file_sha256": diagnostics_digest,
        "prediction_diagnostics_internal_sha256": internal_diagnostics_digest,
        "oof_preregistration_sha256": preregistration_sha256,
        "fold_artifacts": fold_artifacts,
        "guard_enabled": True,
        "fresh_confirmation_policy": "one_shot_fresh50_only",
    }


__all__ = [
    "OOF_TEST_POLICY",
    "validate_authorized_oof_final",
]
