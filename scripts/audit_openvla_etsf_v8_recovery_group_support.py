#!/usr/bin/env python3
"""Audit recovery support at the independent logical-group level.

The v8 evaluator historically reported row counts.  Candidate and continuation
rows from one logical group are correlated and therefore cannot be used to
claim independent class support.  This development-only audit counts a group
once per class and checks whether a five-fold, ten-groups-per-class gate is
even mathematically feasible.  It never opens rollout/HDF5/Fresh data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT = "etsf_v8_recovery_unique_group_support_audit_v1"
FOLD_COUNT = 5
DEFAULT_MINIMUM_CLASS_GROUPS_PER_FOLD = 10
DEFAULT_RECOMMENDED_CLASS_GROUPS = 60


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_fresh(path: Path) -> Path:
    resolved = path.resolve()
    if "fresh" in str(resolved).lower():
        raise ValueError("recovery support audit refuses Fresh paths")
    return resolved


def _vector(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or (length is not None and len(result) != length):
        raise ValueError(f"{name} must be a one-dimensional aligned array")
    return result


def _binary(value: Any, *, name: str, length: int) -> np.ndarray:
    result = _vector(value, name=name, length=length)
    if not np.isfinite(result.astype(np.float64)).all() or np.any(
        (result != 0) & (result != 1)
    ):
        raise ValueError(f"{name} must contain finite binary values")
    return result.astype(bool)


def audit_recovery_group_support(
    arrays: Mapping[str, Any],
    *,
    input_sha256: str,
    minimum_class_groups_per_fold: int = DEFAULT_MINIMUM_CLASS_GROUPS_PER_FOLD,
    recommended_class_groups: int = DEFAULT_RECOMMENDED_CLASS_GROUPS,
) -> dict[str, Any]:
    """Return a signed, row-inflation-resistant recovery support audit."""

    if minimum_class_groups_per_fold <= 0:
        raise ValueError("minimum class groups per fold must be positive")
    required_global = FOLD_COUNT * minimum_class_groups_per_fold
    if recommended_class_groups < required_global:
        raise ValueError("recommended class groups cannot be below the hard minimum")
    if not isinstance(input_sha256, str) or len(input_sha256) != 64:
        raise ValueError("input_sha256 must be a SHA-256 hex digest")
    try:
        int(input_sha256, 16)
    except ValueError as error:
        raise ValueError("input_sha256 must be a SHA-256 hex digest") from error

    groups = _vector(arrays["logical_group"], name="logical group").astype(str)
    length = len(groups)
    if length == 0 or np.any(groups == ""):
        raise ValueError("logical groups must be nonempty")
    folds = _vector(arrays["fold_id"], name="fold id", length=length).astype(
        np.int64
    )
    if set(np.unique(folds).tolist()) != set(range(FOLD_COUNT)):
        raise ValueError("fold ids must cover exactly 0..4")
    regress_mask = _binary(
        arrays["regress_mask"], name="regress mask", length=length
    )
    regress = _binary(
        arrays["regress_label"], name="regress label", length=length
    )
    recovery = _binary(
        arrays["recovery_label"], name="recovery label", length=length
    )
    if np.any(regress_mask & recovery & ~regress):
        raise ValueError("recovery=true must imply regress=true in supervised rows")

    group_fold: dict[str, int] = {}
    for group in sorted(set(groups.tolist())):
        owners = np.unique(folds[groups == group])
        if len(owners) != 1:
            raise ValueError(f"logical group {group!r} crosses outer folds")
        group_fold[group] = int(owners[0])

    conditional = regress_mask & regress

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        positive_groups = set(groups[mask & recovery].tolist())
        negative_groups = set(groups[mask & ~recovery].tolist())
        all_groups = set(groups[mask].tolist())
        return {
            "rows": int(mask.sum()),
            "positive_rows": int((mask & recovery).sum()),
            "negative_rows": int((mask & ~recovery).sum()),
            "logical_groups": len(all_groups),
            "positive_logical_groups": len(positive_groups),
            "negative_logical_groups": len(negative_groups),
            "both_class_logical_groups": len(positive_groups & negative_groups),
            "positive_only_logical_groups": len(positive_groups - negative_groups),
            "negative_only_logical_groups": len(negative_groups - positive_groups),
        }

    by_fold: dict[str, dict[str, Any]] = {}
    for fold in range(FOLD_COUNT):
        row = summarize(conditional & (folds == fold))
        row["minimum_per_class"] = int(minimum_class_groups_per_fold)
        row["support_gate"] = bool(
            row["positive_logical_groups"] >= minimum_class_groups_per_fold
            and row["negative_logical_groups"] >= minimum_class_groups_per_fold
        )
        by_fold[str(fold)] = row

    global_support = summarize(conditional)
    positive = int(global_support["positive_logical_groups"])
    negative = int(global_support["negative_logical_groups"])
    necessary = positive >= required_global and negative >= required_global
    current_split_pass = all(row["support_gate"] for row in by_fold.values())
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": (
            "current_split_passes_unique_group_gate"
            if current_split_pass
            else "fail_closed_insufficient_unique_group_support"
        ),
        "scope": "adaptive_development_only_no_fresh",
        "input_arrays_sha256": input_sha256,
        "estimand_unit": "unique_logical_group_presence_per_class",
        "conditional_subset": "regress_mask_and_ground_truth_regress",
        "candidate_or_continuation_rows_count_as_independent_support": False,
        "fold_count": FOLD_COUNT,
        "minimum_class_groups_per_fold": int(minimum_class_groups_per_fold),
        "mathematical_minimum_global_groups_per_class": required_global,
        "recommended_global_groups_per_class": int(recommended_class_groups),
        "global": global_support,
        "by_fold": by_fold,
        "current_split_support_gate": current_split_pass,
        "necessary_global_support_for_any_disjoint_five_fold_split": necessary,
        "minimum_additional_positive_groups": max(required_global - positive, 0),
        "minimum_additional_negative_groups": max(required_global - negative, 0),
        "recommended_additional_positive_groups": max(
            recommended_class_groups - positive, 0
        ),
        "recommended_additional_negative_groups": max(
            recommended_class_groups - negative, 0
        ),
        "stratified_resplitting_can_satisfy_gate_without_new_groups": bool(
            necessary
        ),
        "repeated_crossfit_can_increase_independent_support": False,
        "fresh50_inputs_accepted": False,
        "fresh50_labels_read": False,
    }
    result["audit_sha256"] = _canonical_sha256(result)
    return result


def load_and_audit(
    input_path: Path,
    *,
    expected_input_sha256: str,
    minimum_class_groups_per_fold: int,
    recommended_class_groups: int,
) -> dict[str, Any]:
    path = _reject_fresh(input_path)
    actual_sha = sha256_path(path)
    if actual_sha != expected_input_sha256:
        raise ValueError("recovery arrays SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        result = audit_recovery_group_support(
            arrays,
            input_sha256=actual_sha,
            minimum_class_groups_per_fold=minimum_class_groups_per_fold,
            recommended_class_groups=recommended_class_groups,
        )
    result["input_arrays"] = str(path)
    unsigned = dict(result)
    unsigned.pop("audit_sha256")
    result["audit_sha256"] = _canonical_sha256(unsigned)
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = _reject_fresh(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--minimum-class-groups-per-fold",
        type=int,
        default=DEFAULT_MINIMUM_CLASS_GROUPS_PER_FOLD,
    )
    parser.add_argument(
        "--recommended-class-groups",
        type=int,
        default=DEFAULT_RECOMMENDED_CLASS_GROUPS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_and_audit(
        args.input,
        expected_input_sha256=args.expected_input_sha256,
        minimum_class_groups_per_fold=args.minimum_class_groups_per_fold,
        recommended_class_groups=args.recommended_class_groups,
    )
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "audit_sha256": result["audit_sha256"],
                "fresh50_inputs_accepted": False,
                "fresh50_labels_read": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FORMAT",
    "audit_recovery_group_support",
    "load_and_audit",
    "sha256_path",
]
