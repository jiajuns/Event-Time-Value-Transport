from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_etsf_transfer_protocol import (  # noqa: E402
    ADAPTATION_SIZES,
    AUDIT_FORMAT,
    PROTOCOL_FORMAT,
    RESULT_FORMAT,
    audit_transfer_weights,
    exact_mcnemar_p,
    evaluate_transfer_results,
    freeze_protocol,
    json_sha256,
    paired_bootstrap_ci,
    validate_protocol,
)
from prepare_etsf_transfer_source_core import expand_source_core  # noqa: E402


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _group(
    *, task: str, policy: str, body: str, seed: int, registry: str
) -> dict[str, object]:
    return {
        "task": task,
        "policy": policy,
        "embodiment": body,
        "requested_seed": seed,
        "resolved_seed": seed + 1_000_000,
        "artifact_sha256": _hash(f"{task}:{policy}:{body}:{seed}:{registry}"),
        "registry": registry,
    }


def _checkpoint(path: Path, *, changed_target_row: bool = False, changed_core: bool = False) -> None:
    policy = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    if changed_target_row:
        policy[1] += 3.0
    core = torch.eye(4)
    if changed_core:
        core[0, 0] += 1.0
    torch.save(
        {
            "model": {
                "action_encoder.policy_embedding.weight": policy,
                "action_encoder.body_embedding.weight": torch.ones(1, 4),
                "semantic.proj.weight": core,
            },
            "config": {"num_bodies": 1, "num_policies": 2},
            "contract": {
                "body_to_id": {"piper": 0},
                "policy_to_id": {"openvla": 0, "__reserved__smolvla": 1},
            },
        },
        path,
    )


def _body_checkpoint(path: Path) -> None:
    torch.save(
        {
            "model": {
                "action_encoder.policy_embedding.weight": torch.ones(1, 4),
                "action_encoder.body_embedding.weight": torch.arange(
                    8, dtype=torch.float32
                ).reshape(2, 4),
                "semantic.proj.weight": torch.eye(4),
            },
            "config": {"num_bodies": 2, "num_policies": 1},
            "contract": {
                "body_to_id": {"piper": 0, "__reserved__aloha": 1},
                "policy_to_id": {"openvla": 0},
            },
        },
        path,
    )


def _draft() -> dict[str, object]:
    target_adaptation = [
        _group(
            task="move_can_pot",
            policy="smolvla",
            body="piper",
            seed=10_000 + index,
            registry="transfer_adaptation",
        )
        for index in range(50)
    ]
    target_validation = [
        _group(
            task="move_can_pot",
            policy="smolvla",
            body="piper",
            seed=20_000 + index,
            registry="transfer_validation",
        )
        for index in range(20)
    ]
    target_confirmation = [
        _group(
            task="move_can_pot",
            policy="smolvla",
            body="piper",
            seed=30_000 + index,
            registry="transfer_confirmation",
        )
        for index in range(50)
    ]
    return {
        "format": PROTOCOL_FORMAT,
        "study_id": "smolvla_on_piper_policy_transfer_v1",
        "axis": "policy",
        "source_domain": {
            "policies": ["openvla"],
            "embodiments": ["piper"],
            "tasks": ["move_can_pot"],
        },
        "target_domain": {
            "policy": "smolvla",
            "embodiment": "piper",
            "tasks": ["move_can_pot"],
        },
        "core": {
            "target_embedding": {
                "parameter": "action_encoder.policy_embedding.weight",
                "row": 1,
                "reservation_name": "__reserved__smolvla",
                "status": "preallocated",
            }
        },
        "contracts": {
            "state": {
                "source_id": "openvla-hidden-4096-v1",
                "target_id": "smolvla-prefix-960-v1",
                "mode": "learned_state_adapter",
            },
            "action_effect": {
                "source_id": "openvla-25x14-v1",
                "target_id": "smolvla-50x14-v1",
                "mode": "policy_action_adapter",
            },
            "predicate": {
                "source_id": "event-spec-sha",
                "target_id": "event-spec-sha",
                "mode": "identity_content_addressed",
            },
            "clock": {
                "source_id": "piper-clock-v1",
                "target_id": "piper-clock-v1",
                "mode": "identity_fixed_body",
            },
            "observer": {
                "source_id": "openvla-hidden-observer-v1",
                "target_id": "smolvla-hidden-observer-v1",
                "mode": "actor_hidden_observer",
            },
        },
        "adaptation": {
            "sizes_per_task": list(ADAPTATION_SIZES),
            "primary_size_per_task": 20,
            "shared_core_update": "forbidden",
            "target_credit_assignment": "adapter_supervision_only_no_td",
            "allowed_external_adapters": [
                "state_adapter",
                "policy_action_adapter",
                "state_observer",
            ],
            "allowed_core_row_updates": [
                {
                    "parameter": "action_encoder.policy_embedding.weight",
                    "row": 1,
                }
            ],
        },
        "splits": {
            "source_training": [
                _group(
                    task="move_can_pot",
                    policy="openvla",
                    body="piper",
                    seed=1,
                    registry="source_core_training",
                )
            ],
            "adaptation": target_adaptation,
            "validation": target_validation,
            "confirmation": target_confirmation,
        },
        "split_contract": {
            "unit": "task_policy_embodiment_requested_resolved_seed",
            "adaptation_order": "ascending_preregistered_index_fixed_prefix",
            "minimum_adaptation_per_task": 50,
            "minimum_validation_per_task": 20,
            "minimum_confirmation_per_task": 50,
            "confirmation_access": "once_after_validation_freeze",
        },
        "baselines": [
            "actor",
            "zero_shot_core",
            "target_from_scratch_matched",
            "full_finetune_upper",
            "no_factorization",
            "actor_hidden_observer",
            "privileged_pose_upper_bound",
        ],
        "acceptance": {
            "prediction": "all_structured_heads_beat_frozen_baselines_at_primary_n",
            "uncertainty": "aurc_below_random_at_primary_n",
            "success": "paired_delta_positive_and_ci95_low_nonnegative",
            "harmful_rate_max": 0.10,
            "minimum_changed": 10,
            "minimum_coverage": 0.10,
            "sample_efficiency": "beats_target_from_scratch_matched_at_primary_n",
        },
    }


def _metrics() -> dict[str, float]:
    return {
        "observer_event_macro_f1": 0.72,
        "observer_event_frequency_macro_f1": 0.40,
        "observer_predicate_macro_f1": 0.75,
        "observer_predicate_constant_macro_f1": 0.50,
        "observer_coverage": 0.95,
        "next_event_macro_f1": 0.70,
        "current_event_macro_f1": 0.50,
        "event_frequency_macro_f1": 0.40,
        "success_pr_auc": 0.65,
        "success_brier": 0.15,
        "constant_success_brier": 0.22,
        "success_ece": 0.08,
        "duration_mae": 3.0,
        "event_body_median_duration_mae": 5.0,
        "object_delta_mae": 0.02,
        "zero_delta_mae": 0.05,
        "pair_accuracy": 0.65,
        "uncertainty_aurc": 0.18,
        "random_aurc": 0.30,
    }


def _results(protocol: dict[str, object], audit: dict[str, object]) -> dict[str, object]:
    ci_low, ci_high = paired_bootstrap_ci(episodes=50, helpful=11, harmful=1)
    curve = [
        {
            "n_per_task": n,
            "split": "validation",
            "group_count": 20,
            "trainable_parameters": 130,
            "metrics": _metrics(),
        }
        for n in ADAPTATION_SIZES
    ]
    return {
        "format": RESULT_FORMAT,
        "study_id": protocol["study_id"],
        "protocol_sha256": json_sha256(protocol),
        "weight_audit_sha256": json_sha256(audit),
        "adapters": [
            {
                "kind": "state_adapter",
                "artifact_sha256": _hash("state-adapter"),
                "trainable_parameters": 60,
            },
            {
                "kind": "policy_action_adapter",
                "artifact_sha256": _hash("action-adapter"),
                "trainable_parameters": 40,
            },
            {
                "kind": "state_observer",
                "artifact_sha256": _hash("state-observer"),
                "trainable_parameters": 30,
            },
        ],
        "adaptation_curve": curve,
        "baselines": [
            {"name": "actor", "n_per_task": 0, "trainable_parameters": 0, "success_rate": 0.20},
            {"name": "zero_shot_core", "n_per_task": 0, "trainable_parameters": 0, "success_rate": 0.20},
            {"name": "target_from_scratch_matched", "n_per_task": 20, "trainable_parameters": 1_000, "success_rate": 0.30},
            {"name": "full_finetune_upper", "n_per_task": 20, "trainable_parameters": 10_000, "success_rate": 0.50},
            {"name": "no_factorization", "n_per_task": 20, "trainable_parameters": 500, "success_rate": 0.35},
            {"name": "actor_hidden_observer", "n_per_task": 20, "trainable_parameters": 300, "success_rate": 0.38},
            {"name": "privileged_pose_upper_bound", "n_per_task": 20, "trainable_parameters": 0, "success_rate": 0.48},
        ],
        "confirmation": {
            "split": "confirmation",
            "group_count": 50,
            "episodes": 50,
            "baseline_successes": 10,
            "plugin_successes": 20,
            "changed": 20,
            "helpful": 11,
            "harmful": 1,
            "proposal_count": 25,
            "coverage": 0.4,
            "paired_delta": 0.2,
            "paired_delta_ci95_low": ci_low,
            "paired_delta_ci95_high": ci_high,
            "exact_mcnemar_p": exact_mcnemar_p(helpful=11, harmful=1),
        },
        "deployment_contract": {
            "target_policy": "smolvla",
            "target_embodiment": "piper",
            "guard_frozen_on": "validation",
            "confirmation_access_count": 1,
            "shared_core_immutable_sha256": audit["immutable_shared_core_sha256"],
            "action_ranking_authorized": True,
            "observer_mode": "actor_hidden_observer",
            "observer_artifact_sha256": _hash("state-observer"),
            "privileged_inputs_used": False,
        },
    }


def test_freeze_validate_weight_audit_and_acceptance(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    _checkpoint(before)
    _checkpoint(after, changed_target_row=True)
    protocol = freeze_protocol(_draft(), before)
    validate_protocol(protocol)
    audit = audit_transfer_weights(protocol, before, after)
    assert audit["format"] == AUDIT_FORMAT
    assert audit["authorized"] is True
    decision = evaluate_transfer_results(protocol, audit, _results(protocol, audit))
    assert decision["shared_core_immutable"] is True
    assert decision["prediction_gate_passed"] is True
    assert decision["action_ranking_authorized"] is True
    assert decision["reasons"] == []


def test_policy_transfer_rejects_body_confound(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    _checkpoint(before)
    draft = _draft()
    draft["target_domain"]["embodiment"] = "aloha"  # type: ignore[index]
    with pytest.raises(ValueError, match="hold embodiment fixed"):
        freeze_protocol(draft, before)


def test_embodiment_protocol_requires_body_action_and_clock_adapters(tmp_path: Path) -> None:
    before = tmp_path / "body_before.pt"
    _body_checkpoint(before)
    draft = _draft()
    draft["study_id"] = "openvla_piper_to_aloha_body_transfer_v1"
    draft["axis"] = "embodiment"
    draft["target_domain"] = {
        "policy": "openvla",
        "embodiment": "aloha",
        "tasks": ["move_can_pot"],
    }
    for split in ("adaptation", "validation", "confirmation"):
        for group in draft["splits"][split]:  # type: ignore[index]
            group["policy"] = "openvla"
            group["embodiment"] = "aloha"
    draft["core"] = {
        "target_embedding": {
            "parameter": "action_encoder.body_embedding.weight",
            "row": 1,
            "reservation_name": "__reserved__aloha",
            "status": "preallocated",
        }
    }
    draft["contracts"] = {
        "state": {
            "source_id": "openvla-hidden-4096-v1",
            "target_id": "openvla-hidden-4096-v1",
            "mode": "identity_content_addressed",
        },
        "action_effect": {
            "source_id": "piper-action-effect-v1",
            "target_id": "aloha-action-effect-v1",
            "mode": "body_action_adapter",
        },
        "predicate": {
            "source_id": "event-spec-sha",
            "target_id": "event-spec-sha",
            "mode": "identity_content_addressed",
        },
        "clock": {
            "source_id": "piper-clock-v1",
            "target_id": "aloha-clock-v1",
            "mode": "clock_adapter",
        },
        "observer": {
            "source_id": "openvla-hidden-observer-v1",
            "target_id": "openvla-hidden-observer-aloha-v1",
            "mode": "actor_hidden_observer",
        },
    }
    draft["adaptation"]["allowed_external_adapters"] = [  # type: ignore[index]
        "body_action_adapter",
        "clock_adapter",
        "state_observer",
    ]
    draft["adaptation"]["allowed_core_row_updates"] = [  # type: ignore[index]
        {"parameter": "action_encoder.body_embedding.weight", "row": 1}
    ]
    protocol = freeze_protocol(draft, before)
    validate_protocol(protocol)
    assert protocol["axis"] == "embodiment"


def test_protocol_rejects_reuse_of_existing_sealed_registry(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    _checkpoint(before)
    draft = _draft()
    draft["splits"]["confirmation"][0]["registry"] = "fresh_confirmation"  # type: ignore[index]
    with pytest.raises(ValueError, match="must not reuse"):
        freeze_protocol(draft, before)


def test_protocol_rejects_privileged_observer_as_primary_path(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    _checkpoint(before)
    draft = _draft()
    draft["contracts"]["observer"]["mode"] = (  # type: ignore[index]
        "privileged_simulator_pose_upper_bound"
    )
    with pytest.raises(ValueError, match="upper-bound baseline only"):
        freeze_protocol(draft, before)


def test_raw_post_training_vocabulary_expansion_cannot_be_frozen(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    expanded = tmp_path / "expanded.pt"
    source_manifest = tmp_path / "source_manifest.json"
    source_split = tmp_path / "source_split.json"
    source_manifest.write_text('{"groups":100}', encoding="utf-8")
    source_split.write_text('{"train":[1,2]}', encoding="utf-8")
    # A true one-policy parent has no reserved row; deterministic preparation
    # alone must not be accepted as a strict source core.
    torch.save(
        {
            "model": {
                "action_encoder.policy_embedding.weight": torch.ones(1, 4),
                "action_encoder.body_embedding.weight": torch.ones(1, 4),
                "semantic.proj.weight": torch.eye(4),
            },
            "config": {"num_bodies": 1, "num_policies": 1},
            "contract": {
                "body_to_id": {"piper": 0},
                "policy_to_id": {"openvla": 0},
            },
        },
        source,
    )
    expand_source_core(
        source,
        expanded,
        axis="policy",
        target_name="smolvla",
        source_manifest=source_manifest,
        source_split=source_split,
    )
    with pytest.raises(ValueError, match="source-only retraining"):
        freeze_protocol(_draft(), expanded)


def test_weight_audit_rejects_shared_core_change(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    _checkpoint(before)
    _checkpoint(after, changed_target_row=True, changed_core=True)
    protocol = freeze_protocol(_draft(), before)
    with pytest.raises(ValueError, match="shared core parameter changed"):
        audit_transfer_weights(protocol, before, after)


def test_failed_confirmation_cannot_be_marked_authorized(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    _checkpoint(before)
    _checkpoint(after, changed_target_row=True)
    protocol = freeze_protocol(_draft(), before)
    audit = audit_transfer_weights(protocol, before, after)
    results = _results(protocol, audit)
    confirmation = results["confirmation"]
    confirmation["plugin_successes"] = 10  # type: ignore[index]
    confirmation["helpful"] = 5  # type: ignore[index]
    confirmation["harmful"] = 5  # type: ignore[index]
    confirmation["paired_delta"] = 0.0  # type: ignore[index]
    ci_low, ci_high = paired_bootstrap_ci(episodes=50, helpful=5, harmful=5)
    confirmation["paired_delta_ci95_low"] = ci_low  # type: ignore[index]
    confirmation["paired_delta_ci95_high"] = ci_high  # type: ignore[index]
    confirmation["exact_mcnemar_p"] = exact_mcnemar_p(helpful=5, harmful=5)  # type: ignore[index]
    results["deployment_contract"]["action_ranking_authorized"] = False  # type: ignore[index]
    decision = evaluate_transfer_results(protocol, audit, results)
    assert decision["action_ranking_authorized"] is False
    assert "paired_success_improvement_not_confirmed" in decision["reasons"]
    assert "not_better_than_target_from_scratch_matched" in decision["reasons"]


def test_paired_ci_is_recomputed_from_episode_discordance(tmp_path: Path) -> None:
    before = tmp_path / "before.pt"
    after = tmp_path / "after.pt"
    _checkpoint(before)
    _checkpoint(after, changed_target_row=True)
    protocol = freeze_protocol(_draft(), before)
    audit = audit_transfer_weights(protocol, before, after)
    results = _results(protocol, audit)
    results["confirmation"]["paired_delta_ci95_low"] += 0.01  # type: ignore[index]
    with pytest.raises(ValueError, match="frozen episode bootstrap"):
        evaluate_transfer_results(protocol, audit, results)
