import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_robotwin2_five_body_postformal_ablation_v1.py"
SPEC = importlib.util.spec_from_file_location("postformal_ablation_watcher", SCRIPT)
watcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watcher)


def test_upstream_requires_the_full_formal_paired_result(tmp_path: Path) -> None:
    report = tmp_path / "paired_success_report.json"
    report.write_text("{}\n", encoding="utf-8")
    value = {
        "format": watcher.UPSTREAM_FORMAT,
        "status": "complete",
        "completed_pairs": 1000,
        "completed_rollouts": 2000,
        "paired_success_report": str(report),
        "paired_success_report_file_sha256": watcher.sha256_file(report),
    }
    watcher.validate_upstream(value)
    value["completed_pairs"] = 20
    with pytest.raises(watcher.PostformalAblationError):
        watcher.validate_upstream(value)


def test_complete_summary_is_fixed_full_budget(tmp_path: Path) -> None:
    binding_sha = "a" * 64
    summary = tmp_path / "offline_ablation_summary.json"
    value = {
        "format": watcher.SUMMARY_FORMAT,
        "status": watcher.SUMMARY_STATUS,
        "binding_file_sha256": binding_sha,
        "inventory": {"decisions": 2000, "branches": 8000},
        "fixed_budget": {
            "variants": list(watcher.EXPECTED_VARIANTS),
            "folds_per_variant": 5,
            "steps_per_member": 3000,
            "ensemble_seeds": list(watcher.EXPECTED_ENSEMBLE_SEEDS),
            "heldout_labels_used_for_checkpoint_selection": False,
            "all_checkpoints_selected_before_any_heldout_payload_open": True,
            "variant_selection_performed": False,
        },
    }
    summary.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.validate_complete_summary(summary, binding_sha) == value
    value["fixed_budget"]["steps_per_member"] = 30
    summary.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(watcher.PostformalAblationError):
        watcher.validate_complete_summary(summary, binding_sha)


def test_runner_command_has_no_reduced_budget_switch(tmp_path: Path) -> None:
    command = watcher.runner_command(tmp_path, "b" * 64)
    assert command[0] == str(watcher.TRAINING_PYTHON)
    assert command[-2:] == ["--python-executable", str(watcher.TRAINING_PYTHON)]
    assert not ({"--steps", "--folds", "--variants", "--limit"} & set(command))


def test_binding_requires_all_five_body_manifests(tmp_path: Path) -> None:
    binding = tmp_path / "binding.json"
    value = {
        "format": watcher.BINDING_FORMAT,
        "heldout_labels_may_train_fit_calibrate_or_select": False,
        "canonical_shared_body_rows": 1,
        "body_manifests": {f"body-{index}": {} for index in range(5)},
    }
    binding.write_text(json.dumps(value), encoding="utf-8")
    assert watcher.validate_binding(binding) == watcher.sha256_file(binding)
    value["body_manifests"].pop("body-4")
    binding.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(watcher.PostformalAblationError):
        watcher.validate_binding(binding)
