from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_openvla_etsf_v8_recovery_group_support as audit


def _arrays(*, groups_per_class_per_fold: int = 2):
    groups = []
    folds = []
    regress = []
    recovery = []
    for fold in range(5):
        for label in (0, 1):
            for index in range(groups_per_class_per_fold):
                group = f"f{fold}-y{label}-g{index}"
                # Two correlated rows must still count as one group.
                groups.extend([group, group])
                folds.extend([fold, fold])
                regress.extend([1, 1])
                recovery.extend([label, label])
    length = len(groups)
    return {
        "logical_group": np.asarray(groups),
        "fold_id": np.asarray(folds),
        "regress_mask": np.ones(length, dtype=np.int8),
        "regress_label": np.asarray(regress, dtype=np.int8),
        "recovery_label": np.asarray(recovery, dtype=np.int8),
    }


def test_counts_unique_groups_and_fails_infeasible_support():
    result = audit.audit_recovery_group_support(
        _arrays(groups_per_class_per_fold=2), input_sha256="a" * 64
    )
    assert result["global"]["positive_rows"] == 20
    assert result["global"]["positive_logical_groups"] == 10
    assert result["global"]["negative_logical_groups"] == 10
    assert result["necessary_global_support_for_any_disjoint_five_fold_split"] is False
    assert result["minimum_additional_positive_groups"] == 40
    assert result["minimum_additional_negative_groups"] == 40
    assert result["current_split_support_gate"] is False


def test_passes_only_when_each_fold_has_ten_groups_per_class():
    result = audit.audit_recovery_group_support(
        _arrays(groups_per_class_per_fold=10), input_sha256="b" * 64
    )
    assert result["necessary_global_support_for_any_disjoint_five_fold_split"] is True
    assert result["current_split_support_gate"] is True
    assert all(row["support_gate"] for row in result["by_fold"].values())


def test_rejects_group_crossing_outer_folds():
    arrays = _arrays()
    first_fold_one_row = np.flatnonzero(arrays["fold_id"] == 1)[0]
    arrays["logical_group"][0] = arrays["logical_group"][first_fold_one_row]
    with pytest.raises(ValueError, match="crosses outer folds"):
        audit.audit_recovery_group_support(arrays, input_sha256="c" * 64)


def test_rejects_recovery_without_regress():
    arrays = _arrays()
    positive = np.flatnonzero(arrays["recovery_label"] == 1)[0]
    arrays["regress_label"][positive] = 0
    with pytest.raises(ValueError, match="imply regress"):
        audit.audit_recovery_group_support(arrays, input_sha256="d" * 64)


def test_load_requires_hash_and_refuses_fresh_path(tmp_path):
    path = tmp_path / "arrays.npz"
    np.savez_compressed(path, **_arrays())
    digest = audit.sha256_path(path)
    result = audit.load_and_audit(
        path,
        expected_input_sha256=digest,
        minimum_class_groups_per_fold=10,
        recommended_class_groups=60,
    )
    assert result["input_arrays_sha256"] == digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.load_and_audit(
            path,
            expected_input_sha256="0" * 64,
            minimum_class_groups_per_fold=10,
            recommended_class_groups=60,
        )
    fresh = tmp_path / "Fresh50_arrays.npz"
    np.savez_compressed(fresh, **_arrays())
    with pytest.raises(ValueError, match="refuses Fresh"):
        audit.load_and_audit(
            fresh,
            expected_input_sha256=audit.sha256_path(fresh),
            minimum_class_groups_per_fold=10,
            recommended_class_groups=60,
        )
