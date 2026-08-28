from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_counterfactual_oof_v6 as launcher  # noqa: E402
import openvla_etsf_counterfactual_oof_v6 as protocol  # noqa: E402
import train_openvla_etsf_counterfactual_oof_v6 as runner  # noqa: E402
from openvla_etsf_event_world_model import (  # noqa: E402
    ActionConditionedEventWorldModel,
    EventWorldModelConfig,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data=tmp_path / "dev250",
        pretrained=tmp_path / "factual.pt",
        event_spec=tmp_path / "events.json",
        output=tmp_path / "out",
        trainer=SCRIPTS / "train_openvla_etsf_counterfactual_oof_v6.py",
        python_bin=Path(sys.executable),
        gpu_index=0,
        gpu_lock=tmp_path / "gpu.lock",
        num_workers=0,
        dry_run=False,
    )


def test_launcher_has_no_final_or_fresh_stage(tmp_path: Path) -> None:
    commands = launcher.build_stage_commands(_args(tmp_path))
    assert [row["stage"] for row in commands] == [
        "preregister", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "select"
    ]
    argv = " ".join(part for row in commands for part in row["argv"])
    assert "fresh" not in argv.lower()
    assert all(row["uses_gpu"] == row["stage"].startswith("fold_") for row in commands)


def test_training_contract_is_low_capacity_fixed_100_steps() -> None:
    args = argparse.Namespace(num_workers=0)
    value = runner._training_args(args)
    assert value.freeze_factual_core is True
    assert value.learning_rate == 1e-3
    assert value.weight_decay == 0.1
    assert value.steps == value.eval_every == 100
    assert value.pairwise_weight == value.baseline_contrast_weight == 1.0
    for name in (
        "success_weight", "outcome_weight", "listwise_weight", "group_centered_weight",
        "event_weight", "duration_weight", "object_weight", "latent_weight",
    ):
        assert getattr(value, name) == 0.0


def test_tensor_digest_ignores_only_rank_head() -> None:
    state = {"core.weight": torch.tensor([1.0]), "action_rank_head.0.weight": torch.tensor([2.0])}
    changed_head = {**state, "action_rank_head.0.weight": torch.tensor([9.0])}
    changed_core = {**state, "core.weight": torch.tensor([9.0])}
    assert runner._tensor_sha(state, core_only=True) == runner._tensor_sha(changed_head, core_only=True)
    assert runner._tensor_sha(state, core_only=True) != runner._tensor_sha(changed_core, core_only=True)


def test_checkpoint_audit_detects_any_factual_mutation(tmp_path: Path) -> None:
    config = EventWorldModelConfig(
        semantic_dim=96, action_rank_residual=True, action_rank_success_only=True
    )
    model = ActionConditionedEventWorldModel(config)
    factual = {"model": {name: value.clone() for name, value in model.state_dict().items()
                         if not name.startswith("action_rank_head.")}}
    contract = {"action_rank_optimization": {"factual_core_trainable_parameters": 0}}
    path = tmp_path / "member.pt"
    torch.save({"model": model.state_dict(), "contract": contract}, path)
    audit = runner._audit_checkpoint(path, factual, config)
    assert audit["factual_core_bit_exact"] is True
    assert audit["trainable_parameter_names"] == ["action_rank_head.0.weight"]
    assert audit["trainable_parameter_count"] == 192
    state = model.state_dict()
    key = next(name for name in state if not name.startswith("action_rank_head."))
    state[key] = state[key].clone()
    state[key].view(-1)[0] += 1
    torch.save({"model": state, "contract": contract}, path)
    with pytest.raises(RuntimeError, match="not bit-exact"):
        runner._audit_checkpoint(path, factual, config)


def test_validate_select_fails_if_fresh_is_authorized(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    value = {
        "format": protocol.SELECTION_FORMAT,
        "status": "complete_development_only",
        "score_ablations": {
            "frozen_base_success_only": {}, "residual_only": {}, "frozen_base_plus_residual": {}
        },
        "authorization": {"authorized": True, "fresh_confirmation_allowed": True},
    }
    value["selection_sha256"] = protocol.canonical_sha256(value)
    (output / "oof_selection_v6.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="never authorize"):
        launcher.validate_stage("select", output)


def test_score_report_contains_all_required_decomposition_metrics() -> None:
    rows = [{
        "baseline_index": 0,
        "success": np.asarray([0.0, 1.0, 0.0, 0.0]),
        "frozen_base_success_only": np.asarray([0.0, 1.0, 0.0, 0.0]),
        "residual_only": np.asarray([0.0, -1.0, 0.0, 0.0]),
        "frozen_base_plus_residual": np.asarray([0.0, 0.0, 0.0, 0.0]),
    }]
    assert runner._score_report(rows, "frozen_base_success_only")["top1_success"] == 1.0
    assert runner._score_report(rows, "residual_only")["top1_success"] == 0.0
