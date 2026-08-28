#!/usr/bin/env python3
"""Read-only CPU five-fold OOF diagnostic for schema-v5 action signal.

This diagnostic asks a deliberately narrow question: can terminal candidate
success be ranked from the current state and the candidate action delta at all?
It never trains the world model, never reads fresh-confirmation data, and never
changes candidate labels.  Three preregistered linear baselines are reported;
none is selected using OOF outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from openvla_etsf_counterfactual_oof import (
        make_oof_folds as _shared_make_oof_folds,
        validate_oof_folds as _shared_validate_oof_folds,
    )
except ImportError:
    _shared_make_oof_folds = None
    _shared_validate_oof_folds = None


FORMAT = "etsf_schema5_action_signal_five_fold_oof_v1"
VARIANTS = (
    "action_delta_logistic",
    "state_action_interaction_logistic",
    "state_action_interaction_pairwise",
)
PRIMARY_VARIANT = "state_action_interaction_pairwise"
STATE_PCA_COMPONENTS = 8
ACTION_PCA_COMPONENTS = 8
LOGISTIC_C = 1.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260903
OOF_SPLIT_SEED = 20260827


def _embedded_oof_folds(logical_keys: Sequence[str]) -> dict[str, Any]:
    """Dependency-free copy of the frozen identity-only fold assignment."""

    keys = sorted(map(str, logical_keys))
    if len(keys) != 100 or len(set(keys)) != 100:
        raise RuntimeError("formal OOF requires exactly 100 unique logical groups")
    ordered = sorted(
        keys,
        key=lambda key: hashlib.sha256(
            f"{OOF_SPLIT_SEED}|{key}".encode("utf-8")
        ).hexdigest(),
    )
    folds = []
    for fold_id in range(5):
        holdout = sorted(ordered[fold_id * 20 : (fold_id + 1) * 20])
        holdout_set = set(holdout)
        folds.append(
            {
                "fold_id": fold_id,
                "training_groups": [key for key in keys if key not in holdout_set],
                "oof_holdout_groups": holdout,
                "checkpoint_selection": "fixed_final_step_no_holdout_early_stop",
            }
        )
    payload = {
        "format": "etsf_counterfactual_five_fold_oof_v1",
        "split_algorithm": "sha256_sort_contiguous_equal_folds_v1",
        "split_seed": OOF_SPLIT_SEED,
        "development_groups": keys,
        "folds": folds,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["preregistration_sha256"] = hashlib.sha256(encoded).hexdigest()
    _validate_embedded_oof_folds(payload, keys)
    return payload


def _validate_embedded_oof_folds(
    manifest: Mapping[str, Any], logical_keys: Sequence[str]
) -> None:
    keys = set(map(str, logical_keys))
    folds = manifest.get("folds")
    if len(keys) != 100 or not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("embedded OOF fold contract is incomplete")
    owners: set[str] = set()
    for fold_id, fold in enumerate(folds):
        train = set(map(str, fold.get("training_groups", [])))
        holdout = set(map(str, fold.get("oof_holdout_groups", [])))
        if int(fold.get("fold_id", -1)) != fold_id:
            raise RuntimeError("embedded OOF fold ids changed")
        if len(train) != 80 or len(holdout) != 20:
            raise RuntimeError("embedded OOF fold size changed")
        if train & holdout or train | holdout != keys or owners & holdout:
            raise RuntimeError("embedded OOF train/holdout leakage or duplication")
        owners.update(holdout)
    if owners != keys:
        raise RuntimeError("embedded OOF holdout coverage is incomplete")


def make_diagnostic_oof_folds(logical_keys: Sequence[str]) -> dict[str, Any]:
    embedded = _embedded_oof_folds(logical_keys)
    if _shared_make_oof_folds is None:
        return embedded
    shared = _shared_make_oof_folds(logical_keys)
    embedded_assignments = [
        (fold["training_groups"], fold["oof_holdout_groups"])
        for fold in embedded["folds"]
    ]
    shared_assignments = [
        (fold["training_groups"], fold["oof_holdout_groups"])
        for fold in shared["folds"]
    ]
    if embedded_assignments != shared_assignments:
        raise RuntimeError("embedded diagnostic folds diverged from shared OOF folds")
    return shared


def validate_diagnostic_oof_folds(
    manifest: Mapping[str, Any], logical_keys: Sequence[str]
) -> None:
    if _shared_validate_oof_folds is not None and "status" in manifest:
        _shared_validate_oof_folds(manifest, logical_keys)
    else:
        _validate_embedded_oof_folds(manifest, logical_keys)


@dataclass(frozen=True)
class DiagnosticGroup:
    logical_key: str
    seed: int
    path: str
    hidden: np.ndarray
    actions: np.ndarray
    success: np.ndarray
    candidate_distance: np.ndarray
    candidate_names: tuple[str, ...]
    baseline_index: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decode(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode() if isinstance(value, bytes) else str(value) for value in values
    )


def _resolve_group_path(root: Path, recorded: str) -> Path:
    recorded_path = Path(recorded)
    candidates = (
        (recorded_path,) if recorded_path.is_absolute() else ()
    ) + (root / "groups" / recorded_path, root / recorded_path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"candidate group path escapes the schema-v5 root: {recorded}"
            ) from error
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"candidate group is unavailable: {recorded}")


def load_schema5_development(root: Path) -> tuple[list[DiagnosticGroup], dict[str, Any]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise RuntimeError("schema-v5 manifest must contain a JSON object")
    if manifest.get("status") != "complete" or int(
        manifest.get("schema_version", -1)
    ) != 5:
        raise RuntimeError("action diagnostic requires a complete schema-v5 root")
    if manifest.get("seed_registry") == "explicit_fresh_confirmation" or manifest.get(
        "fresh_seed_manifest_sha256"
    ) not in (None, ""):
        raise RuntimeError("fresh confirmation data is forbidden in this diagnostic")
    if manifest.get("language_contract") != (
        "same_instruction_for_initial_query_and_all_candidate_branches"
    ):
        raise RuntimeError("candidate root lacks the fixed-language contract")
    rows = manifest.get("groups")
    if not isinstance(rows, list) or len(rows) != 100:
        raise RuntimeError("formal action diagnostic requires exactly 100 groups")
    groups = []
    file_audit = []
    for item in rows:
        if not isinstance(item, Mapping) or not item.get("path"):
            raise RuntimeError("candidate manifest contains an invalid group row")
        path = _resolve_group_path(root, str(item["path"]))
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != 5:
                raise RuntimeError(f"non-schema-v5 candidate group: {path}")
            if not bool(handle.attrs.get("branch_instruction_consistent", False)):
                raise RuntimeError(f"candidate language changed: {path}")
            task = str(handle.attrs.get("task", manifest.get("task", "unknown")))
            body = str(handle.attrs.get("body", manifest.get("body", "unknown")))
            seed = int(handle.attrs.get("resolved_seed", handle.attrs.get("seed", -1)))
            hidden = handle["initial_hidden"][:].astype(np.float32)
            actions = handle["candidate_actions"][:].astype(np.float32)
            success = handle["success"][:].astype(np.float32)
            names = _decode(handle["candidate_names"][:])
            distance = (
                handle["normalized_l2_from_baseline"][:].astype(np.float32)
                if "normalized_l2_from_baseline" in handle
                else np.sqrt(
                    np.mean(
                        np.square(actions - actions[0:1]), axis=(1, 2)
                    )
                ).astype(np.float32)
            )
        if seed < 0:
            raise RuntimeError(f"candidate group lacks resolved seed: {path}")
        if hidden.ndim != 1 or actions.ndim != 3:
            raise RuntimeError(f"invalid state/action shape: {path}")
        if success.shape != (len(actions),) or distance.shape != success.shape:
            raise RuntimeError(f"candidate fields are misaligned: {path}")
        if len(names) != len(actions) or names.count("deterministic") != 1:
            raise RuntimeError(f"candidate baseline contract is invalid: {path}")
        if not all(
            np.isfinite(value).all() for value in (hidden, actions, success, distance)
        ):
            raise RuntimeError(f"candidate group contains non-finite values: {path}")
        if np.any((success < 0) | (success > 1)):
            raise RuntimeError(f"success labels are outside [0,1]: {path}")
        logical_key = f"{task}|{body}|{seed}"
        groups.append(
            DiagnosticGroup(
                logical_key=logical_key,
                seed=seed,
                path=str(path),
                hidden=hidden,
                actions=actions,
                success=success,
                candidate_distance=distance,
                candidate_names=names,
                baseline_index=names.index("deterministic"),
            )
        )
        digest = sha256(path)
        recorded_digest = str(item.get("sha256", ""))
        if recorded_digest and recorded_digest != digest:
            raise RuntimeError(f"candidate HDF5 SHA mismatch: {path}")
        file_audit.append(
            {"logical_key": logical_key, "path": str(path), "sha256": digest}
        )
    keys = [group.logical_key for group in groups]
    seeds = [group.seed for group in groups]
    if len(set(keys)) != 100 or len(set(seeds)) != 100:
        raise RuntimeError("candidate development root has duplicate groups/scenes")
    candidate_counts = {len(group.actions) for group in groups}
    action_shapes = {group.actions.shape[1:] for group in groups}
    hidden_dims = {len(group.hidden) for group in groups}
    if len(candidate_counts) != 1 or len(action_shapes) != 1 or len(hidden_dims) != 1:
        raise RuntimeError("candidate feature contracts differ across groups")
    groups.sort(key=lambda group: group.logical_key)
    return groups, {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "groups": len(groups),
        "candidate_count": next(iter(candidate_counts)),
        "action_shape": list(next(iter(action_shapes))),
        "hidden_dim": next(iter(hidden_dims)),
        "group_files": sorted(file_audit, key=lambda row: row["logical_key"]),
        "labels_read": "schema5_train100_development_success_only",
        "fresh_confirmation_labels_read": False,
    }


def _action_stats(
    actions: np.ndarray,
    baseline: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    normalized = (actions - action_mean[None, None]) / action_std[None, None]
    delta = (actions - baseline[None]) / action_std[None, None]
    velocity = np.diff(normalized, axis=1)
    parts = [
        normalized.mean(1),
        normalized.std(1),
        normalized[:, 0],
        normalized[:, -1],
        delta.mean(1),
        delta.std(1),
        np.abs(delta).mean(1),
        np.abs(delta).max(1),
        delta[:, 0],
        delta[:, -1],
        np.abs(velocity).mean(1),
        np.abs(velocity).max(1),
        np.sqrt(np.mean(np.square(delta), axis=(1, 2)))[:, None],
        np.max(np.abs(delta), axis=(1, 2))[:, None],
        distance[:, None],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


class FoldFeatures:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, groups: Sequence[DiagnosticGroup]) -> "FoldFeatures":
        action_rows = np.concatenate([group.actions for group in groups])
        self.action_mean = action_rows.mean((0, 1))
        self.action_std = np.maximum(action_rows.std((0, 1)), 1e-3)
        raw_action = np.concatenate([self.raw_action(group) for group in groups])
        self.action_scaler = StandardScaler().fit(raw_action)
        action_z = self.action_scaler.transform(raw_action)
        action_components = min(
            ACTION_PCA_COMPONENTS, action_z.shape[0] - 1, action_z.shape[1]
        )
        self.action_pca = PCA(
            n_components=action_components,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(action_z)

        hidden = np.stack([group.hidden for group in groups]).astype(np.float64)
        self.state_scaler = StandardScaler().fit(hidden)
        hidden_z = self.state_scaler.transform(hidden)
        state_components = min(
            STATE_PCA_COMPONENTS, hidden_z.shape[0] - 1, hidden_z.shape[1]
        )
        self.state_pca = PCA(
            n_components=state_components,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(hidden_z)
        combined = self._combined(groups)
        self.combined_scaler = StandardScaler().fit(combined)
        return self

    def raw_action(self, group: DiagnosticGroup) -> np.ndarray:
        return _action_stats(
            group.actions,
            group.actions[group.baseline_index],
            self.action_mean,
            self.action_std,
            group.candidate_distance,
        )

    def action(self, groups: Sequence[DiagnosticGroup]) -> np.ndarray:
        return self.action_scaler.transform(
            np.concatenate([self.raw_action(group) for group in groups])
        )

    def _combined(self, groups: Sequence[DiagnosticGroup]) -> np.ndarray:
        action = self.action(groups)
        action_pc = self.action_pca.transform(action)
        state_group = self.state_pca.transform(
            self.state_scaler.transform(
                np.stack([group.hidden for group in groups]).astype(np.float64)
            )
        )
        state = np.concatenate(
            [
                np.repeat(state_group[index : index + 1], len(group.actions), axis=0)
                for index, group in enumerate(groups)
            ]
        )
        interaction = np.einsum("ni,nj->nij", state, action_pc).reshape(
            len(action), -1
        )
        return np.concatenate([action, state, interaction], axis=1)

    def combined(self, groups: Sequence[DiagnosticGroup]) -> np.ndarray:
        return self.combined_scaler.transform(self._combined(groups))


def _labels(groups: Sequence[DiagnosticGroup]) -> np.ndarray:
    return np.concatenate([group.success for group in groups]).astype(np.int64)


def _slices(groups: Sequence[DiagnosticGroup]) -> list[slice]:
    result = []
    offset = 0
    for group in groups:
        result.append(slice(offset, offset + len(group.actions)))
        offset += len(group.actions)
    return result


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> Any:
    if len(np.unique(labels)) < 2:
        return None
    return LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=20260903,
    ).fit(features, labels)


def _fit_pairwise(
    features: np.ndarray,
    groups: Sequence[DiagnosticGroup],
) -> Any:
    differences = []
    labels = []
    for group, group_slice in zip(groups, _slices(groups)):
        x = features[group_slice]
        y = group.success
        for positive in np.flatnonzero(y > 0.5):
            for negative in np.flatnonzero(y < 0.5):
                difference = x[positive] - x[negative]
                differences.extend([difference, -difference])
                labels.extend([1, 0])
    if not differences:
        return None
    return LogisticRegression(
        C=LOGISTIC_C,
        class_weight=None,
        fit_intercept=False,
        solver="liblinear",
        max_iter=2000,
        random_state=20260903,
    ).fit(np.asarray(differences), np.asarray(labels))


def _score(model: Any, features: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(features), dtype=np.float64)
    return model.decision_function(features).astype(np.float64)


def _select_with_actor_tie(score: np.ndarray, baseline: int) -> int:
    maximum = float(np.max(score))
    tied = np.flatnonzero(np.isclose(score, maximum, rtol=0.0, atol=1e-12))
    if baseline in tied:
        return baseline
    return int(tied[0])


def _exact_sign_p(helpful: int, harmful: int) -> float:
    count = helpful + harmful
    if count == 0:
        return 1.0
    return float(
        sum(math.comb(count, value) for value in range(helpful, count + 1))
        / 2**count
    )


def _bootstrap_ci(delta: np.ndarray) -> list[float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta))
    )
    means = delta[indices].mean(1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def summarize_oof(
    groups: Sequence[DiagnosticGroup], scores: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    result = {}
    flat_labels = _labels(groups)
    slices = _slices(groups)
    actor = np.asarray(
        [group.success[group.baseline_index] for group in groups], dtype=np.float64
    )
    oracle = np.asarray([group.success.max() for group in groups], dtype=np.float64)
    for variant in VARIANTS:
        variant_scores = np.asarray(scores[variant], dtype=np.float64)
        selected_success = []
        selected_indices = []
        pair_correct = 0
        pair_total = 0
        for group, group_slice in zip(groups, slices):
            group_score = variant_scores[group_slice]
            selected = _select_with_actor_tie(group_score, group.baseline_index)
            selected_indices.append(selected)
            selected_success.append(float(group.success[selected]))
            for positive in np.flatnonzero(group.success > 0.5):
                for negative in np.flatnonzero(group.success < 0.5):
                    pair_total += 1
                    pair_correct += int(group_score[positive] > group_score[negative])
        selected = np.asarray(selected_success)
        delta = selected - actor
        helpful = int((delta > 0).sum())
        harmful = int((delta < 0).sum())
        try:
            auc = float(roc_auc_score(flat_labels, variant_scores))
        except ValueError:
            auc = None
        result[variant] = {
            "groups": len(groups),
            "actor_baseline_success_rate": float(actor.mean()),
            "selected_success_rate": float(selected.mean()),
            "oracle_success_rate": float(oracle.mean()),
            "oracle_headroom_groups": int((oracle > actor).sum()),
            "paired_success_delta": float(delta.mean()),
            "paired_group_bootstrap_95_ci": _bootstrap_ci(delta),
            "changed_from_actor": int(
                sum(
                    selected_index != group.baseline_index
                    for selected_index, group in zip(selected_indices, groups)
                )
            ),
            "helpful_changes": helpful,
            "harmful_changes": harmful,
            "outcome_tied_changes": int(
                sum(
                    selected_index != group.baseline_index and change == 0
                    for selected_index, group, change in zip(
                        selected_indices, groups, delta
                    )
                )
            ),
            "exact_one_sided_sign_mcnemar_p": _exact_sign_p(helpful, harmful),
            "candidate_auc": auc,
            "within_group_success_pair_accuracy": (
                pair_correct / pair_total if pair_total else None
            ),
            "within_group_success_pairs": pair_total,
        }
    return result


def run_oof(groups: Sequence[DiagnosticGroup]) -> dict[str, Any]:
    fold_manifest = make_diagnostic_oof_folds(
        [group.logical_key for group in groups]
    )
    validate_diagnostic_oof_folds(
        fold_manifest, [group.logical_key for group in groups]
    )
    by_key = {group.logical_key: group for group in groups}
    score_parts = {variant: {} for variant in VARIANTS}
    fold_audits = []
    for fold in fold_manifest["folds"]:
        fold_id = int(fold["fold_id"])
        train = [by_key[key] for key in fold["training_groups"]]
        holdout = [by_key[key] for key in fold["oof_holdout_groups"]]
        features = FoldFeatures(20260903 + fold_id).fit(train)
        train_action = features.action(train)
        holdout_action = features.action(holdout)
        train_combined = features.combined(train)
        holdout_combined = features.combined(holdout)
        train_labels = _labels(train)
        models = {
            "action_delta_logistic": _fit_logistic(train_action, train_labels),
            "state_action_interaction_logistic": _fit_logistic(
                train_combined, train_labels
            ),
            "state_action_interaction_pairwise": _fit_pairwise(
                train_combined, train
            ),
        }
        fold_scores = {
            "action_delta_logistic": _score(
                models["action_delta_logistic"], holdout_action
            ),
            "state_action_interaction_logistic": _score(
                models["state_action_interaction_logistic"], holdout_combined
            ),
            "state_action_interaction_pairwise": _score(
                models["state_action_interaction_pairwise"], holdout_combined
            ),
        }
        for variant, values in fold_scores.items():
            offset = 0
            for group in holdout:
                width = len(group.actions)
                score_parts[variant][group.logical_key] = values[offset : offset + width]
                offset += width
        fold_audits.append(
            {
                "fold_id": fold_id,
                "training_groups": len(train),
                "oof_holdout_groups": len(holdout),
                "group_overlap": False,
                "state_pca_components": int(features.state_pca.n_components_),
                "action_pca_components": int(features.action_pca.n_components_),
                "train_candidates": int(sum(len(group.actions) for group in train)),
                "holdout_candidates": int(
                    sum(len(group.actions) for group in holdout)
                ),
            }
        )
    ordered = sorted(groups, key=lambda group: group.logical_key)
    flat_scores = {}
    for variant in VARIANTS:
        if set(score_parts[variant]) != {group.logical_key for group in groups}:
            raise RuntimeError(f"OOF prediction coverage is incomplete: {variant}")
        flat_scores[variant] = np.concatenate(
            [score_parts[variant][group.logical_key] for group in ordered]
        )
    return {
        "fold_protocol": {
            "format": fold_manifest["format"],
            "preregistration_sha256": fold_manifest["preregistration_sha256"],
            "split_algorithm": "sha256_sort_contiguous_equal_folds_v1",
            "split_seed": OOF_SPLIT_SEED,
            "shared_oof_helper_available": _shared_make_oof_folds is not None,
            "fold_count": 5,
            "groups_per_fold": 20,
            "every_group_predicted_once": True,
            "train_holdout_group_leakage": False,
        },
        "fold_audits": fold_audits,
        "metrics": summarize_oof(ordered, flat_scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = args.data.resolve()
    output = args.output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(f"diagnostic output exists; refusing overwrite: {output}")
    try:
        output.relative_to(data)
    except ValueError:
        pass
    else:
        raise RuntimeError("diagnostic output may not be written inside the data root")
    groups, data_audit = load_schema5_development(data)
    result = run_oof(groups)
    payload = {
        "format": FORMAT,
        "status": "complete",
        "purpose": "read_only_action_signal_learnability_diagnostic_not_guard_selection",
        "primary_variant_preregistered": PRIMARY_VARIANT,
        "variants_reported_without_oof_selection": list(VARIANTS),
        "fixed_hyperparameters": {
            "state_pca_components": STATE_PCA_COMPONENTS,
            "action_pca_components": ACTION_PCA_COMPONENTS,
            "logistic_C": LOGISTIC_C,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "data_audit": data_audit,
        **result,
        "fresh_confirmation_labels_read": False,
        "writes_to_data_root": False,
        "gpu_used": False,
    }
    atomic_json(output, payload)
    print("ACTION_SIGNAL_OOF=" + json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
