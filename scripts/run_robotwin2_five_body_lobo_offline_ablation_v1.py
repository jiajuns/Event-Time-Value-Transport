#!/usr/bin/env python3
"""Run the fixed full-8000-branch, five-fold LOBO offline ablation.

This is an explicit training orchestrator, not a small-sample diagnostic.  It
accepts only a binding containing all 2,000 four-candidate decisions, launches
the same five-member/3,000-step budget for every fold and variant, and reports
both source-validation and frozen-checkpoint posthoc heldout metrics.  Heldout
payloads remain unopened until all source-only checkpoint selection finishes
and may never train or automatically select checkpoints/variants.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer


FORMAT = "etsf_robotwin2_five_body_lobo_offline_ablation_v1"
STATUS = "complete_frozen_checkpoint_posthoc_heldout_ablation"
VARIANTS = trainer.ABLATION_VARIANTS
QUERY_INDICES = (0, 5, 10, 15)
SEEDS_PER_CONDITION_QUERY = 50
DECISIONS_PER_BODY = 400
TOTAL_DECISIONS = 2000
TOTAL_BRANCHES = 8000
SPLIT_SEED = 20260901
STEPS_PER_MEMBER = 3000
EVAL_EVERY = 100
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)


class AblationError(RuntimeError):
    """The inventory, budget, fold isolation, or result contract changed."""


def validate_complete_inventory(audit: Mapping[str, Any]) -> dict[str, Any]:
    bodies: dict[str, Any] = {}
    total = 0
    for body in trainer.BODIES:
        groups = audit["manifests"][body]["groups"]
        if len(groups) != DECISIONS_PER_BODY:
            raise AblationError(f"{body} does not contain exactly 400 decisions")
        units: dict[tuple[str, int], set[int]] = defaultdict(set)
        for row in groups:
            condition = str(row.get("condition"))
            query = row.get("root_query_index")
            seed = row.get("requested_seed")
            if (
                condition not in trainer.CONDITIONS
                or isinstance(query, bool)
                or query not in QUERY_INDICES
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed in units[(condition, int(query))]
            ):
                raise AblationError(f"{body} has an invalid condition/query/seed inventory")
            units[(condition, int(query))].add(seed)
        if set(units) != {
            (condition, query)
            for condition in trainer.CONDITIONS
            for query in QUERY_INDICES
        } or any(len(seeds) != SEEDS_PER_CONDITION_QUERY for seeds in units.values()):
            raise AblationError(f"{body} is not complete 2x4x50")
        bodies[body] = {
            "decisions": len(groups),
            "branches": len(groups) * trainer.CANDIDATE_COUNT,
            "condition_query_seed_counts": {
                f"{condition}|query={query}": len(units[(condition, query)])
                for condition in trainer.CONDITIONS
                for query in QUERY_INDICES
            },
        }
        total += len(groups)
    if total != TOTAL_DECISIONS:
        raise AblationError("ablation input is not the complete 2,000/8,000 inventory")
    return {
        "decisions": total,
        "branches": total * trainer.CANDIDATE_COUNT,
        "bodies": bodies,
    }


def fold_command(
    *,
    python_executable: str,
    binding: Path,
    binding_sha256: str,
    output: Path,
    held_out_body: str,
    variant: str,
) -> list[str]:
    return [
        python_executable,
        str(Path(trainer.__file__).resolve()),
        "--mode", "train-fold",
        "--binding", str(binding),
        "--binding-sha256", binding_sha256,
        "--held-out-body", held_out_body,
        "--split-seed", str(SPLIT_SEED),
        "--output", str(output),
        "--device", "cuda",
        "--steps", str(STEPS_PER_MEMBER),
        "--eval-every", str(EVAL_EVERY),
        "--batch-size", str(BATCH_SIZE),
        "--learning-rate", str(LEARNING_RATE),
        "--ablation-variant", variant,
        "--ensemble-seeds", *[str(seed) for seed in ENSEMBLE_SEEDS],
    ]


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        current = current.get(key) if isinstance(current, Mapping) else None
    return current


def _member_mean(members: Sequence[Mapping[str, Any]], *path: str) -> float | None:
    values = [_nested(member["source_validation"], path) for member in members]
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


METRICS = {
    "best_of_4_delta_success_rate": ("candidate_ranking", "macro_delta_success_rate"),
    "selected_success_rate": ("candidate_ranking", "macro_selected_success_rate"),
    "oracle_success_rate": ("candidate_ranking", "macro_oracle_success_rate"),
    "pairwise_accuracy": ("candidate_ranking", "pairwise_accuracy"),
    "success_brier": ("success_brier",),
    "success_auroc": ("success_auroc",),
    "post_event_macro_f1": ("post_event", "macro_f1"),
    "post_event_accuracy": ("post_event", "accuracy"),
    "next_event_macro_f1": ("next_event", "macro_f1"),
    "next_event_accuracy": ("next_event", "accuracy"),
    "duration_observed_mae_seconds": ("observed_duration_mae",),
    "duration_observed_nll": ("observed_duration_nll",),
    "object_rmse": ("object_rmse",),
    "object_nll": ("object_nll",),
}


def summarize_fold(
    summary: Mapping[str, Any], *, held_out_body: str, variant: str
) -> dict[str, Any]:
    members = summary.get("members")
    expected_budget = {
        "steps_per_member": STEPS_PER_MEMBER,
        "eval_every_steps": EVAL_EVERY,
        "batch_size_rows": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "ensemble_members": len(ENSEMBLE_SEEDS),
    }
    if (
        summary.get("status") != "source_only_checkpoint_selection_complete"
        or summary.get("held_out_body") != held_out_body
        or summary.get("ablation") != trainer.ablation_contract(variant)
        or summary.get("candidate_rank_contract")
        != trainer.summary_candidate_rank_contract(variant)
        or summary.get("training_budget") != expected_budget
        or summary.get("heldout_labels_used_for_normalization_training_or_selection") is not False
        or summary.get("heldout_group_npz_opened") != 0
        or summary.get("preflight", {}).get("split_unit")
        != "body_condition_requested_seed_all_queries"
        or not isinstance(members, list)
        or len(members) != len(ENSEMBLE_SEEDS)
        or [member.get("seed") for member in members] != list(ENSEMBLE_SEEDS)
    ):
        raise AblationError(f"{variant}/{held_out_body} fold contract changed")
    return {
        "held_out_body": held_out_body,
        "source_bodies": summary.get("source_bodies"),
        "member_count": len(members),
        "evaluation_role": "source_validation_used_for_checkpoint_selection",
        "metric_aggregation": "arithmetic_mean_of_five_selected_members",
        "metrics": {
            name: _member_mean(members, *path) for name, path in METRICS.items()
        },
    }


class _FrozenRankEnsemble(torch.nn.Module):
    """Match formal inference by averaging the five frozen rank scores."""

    def __init__(
        self, models: Sequence[trainer.EffectAlignedSharedEventHead]
    ) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scores = [model(batch)["candidate_rank_logit"] for model in self.models]
        return {"candidate_rank_logit": torch.stack(scores).mean(dim=0)}


def _load_frozen_members(
    summary: Mapping[str, Any], *, held_out_body: str, variant: str,
    device: torch.device,
) -> list[trainer.EffectAlignedSharedEventHead]:
    models = []
    for expected_member, item in enumerate(summary["members"]):
        checkpoint_path = Path(str(item.get("checkpoint", ""))).expanduser().resolve()
        if (
            item.get("member") != expected_member
            or not checkpoint_path.is_file()
            or trainer.sha256_file(checkpoint_path) != item.get("checkpoint_sha256")
        ):
            raise AblationError(f"{variant}/{held_out_body} selected checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            checkpoint.get("format") != trainer.FORMAT
            or checkpoint.get("member") != expected_member
            or checkpoint.get("seed") != ENSEMBLE_SEEDS[expected_member]
            or checkpoint.get("held_out_body") != held_out_body
            or checkpoint.get("ablation") != trainer.ablation_contract(variant)
            or checkpoint.get("candidate_rank_contract")
            != trainer.checkpoint_candidate_rank_contract(variant)
            or checkpoint.get(
                "heldout_rows_used_for_training_normalization_or_selection"
            ) != 0
        ):
            raise AblationError(f"{variant}/{held_out_body} checkpoint contract changed")
        model = trainer.EffectAlignedSharedEventHead(variant).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def evaluate_posthoc_heldout_fold(
    summary: Mapping[str, Any], audit: Mapping[str, Any], *, held_out_body: str,
    variant: str, device: torch.device,
) -> dict[str, Any]:
    """Read heldout labels only after source-only checkpoints are frozen."""

    summarize_fold(summary, held_out_body=held_out_body, variant=variant)
    models = _load_frozen_members(
        summary, held_out_body=held_out_body, variant=variant, device=device
    )
    groups = audit["manifests"][held_out_body]["groups"]
    if len(groups) != DECISIONS_PER_BODY:
        raise AblationError(f"{held_out_body} heldout inventory changed")
    rows = [
        row
        for group in groups
        for row in trainer._npz_rows(group, body=held_out_body)
    ]
    loader = DataLoader(
        trainer.core.TransitionDataset(rows, {held_out_body: 0}),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=trainer.core.collate_rows,
    )
    ranking = trainer.evaluate_candidate_ranking(
        _FrozenRankEnsemble(models), loader, device
    )
    member_predictions = [
        trainer.core.evaluate_validation_model(model, loader, device)
        for model in models
    ]
    prediction_members = [
        {"source_validation": metrics} for metrics in member_predictions
    ]
    metrics = {
        name: (
            None
            if path[0] == "candidate_ranking" and ranking[path[-1]] is None
            else float(ranking[path[-1]])
            if path[0] == "candidate_ranking"
            else _member_mean(prediction_members, *path)
        )
        for name, path in METRICS.items()
    }
    return {
        "held_out_body": held_out_body,
        "source_bodies": summary.get("source_bodies"),
        "evaluation_role": "posthoc_heldout_only_after_all_checkpoint_selection",
        "heldout_decisions": len(groups),
        "heldout_branches": len(rows),
        "candidate_metric_aggregation": "mean_five_frozen_rank_scores_then_best_of_4",
        "prediction_metric_aggregation": (
            "arithmetic_mean_of_five_member_metrics_not_ensemble_calibrated"
        ),
        "heldout_labels_used_for_training_checkpoint_or_variant_selection": False,
        "metrics": metrics,
    }


def aggregate_variants(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        folds = [
            summarize_fold(
                summaries[variant][body], held_out_body=body, variant=variant
            )
            for body in trainer.BODIES
        ]
        macro = {}
        for name in METRICS:
            values = [fold["metrics"][name] for fold in folds]
            present = [float(value) for value in values if value is not None]
            macro[name] = statistics.fmean(present) if present else None
        result[variant] = {
            "ablation": trainer.ablation_contract(variant),
            "folds": folds,
            "equal_fold_macro": macro,
        }
    baseline = result["success_only"]["equal_fold_macro"]
    result["comparison_to_success_only"] = {
        variant: {
            name: (
                None
                if result[variant]["equal_fold_macro"][name] is None
                or baseline[name] is None
                else result[variant]["equal_fold_macro"][name] - baseline[name]
            )
            for name in METRICS
        }
        for variant in VARIANTS
        if variant != "success_only"
    }
    return result


def aggregate_posthoc_heldout(
    evaluations: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        folds = [evaluations[variant][body] for body in trainer.BODIES]
        macro = {}
        for name in METRICS:
            values = [fold["metrics"][name] for fold in folds]
            present = [float(value) for value in values if value is not None]
            macro[name] = statistics.fmean(present) if present else None
        result[variant] = {
            "ablation": trainer.ablation_contract(variant),
            "folds": folds,
            "equal_fold_macro": macro,
        }
    baseline = result["success_only"]["equal_fold_macro"]
    result["comparison_to_success_only"] = {
        variant: {
            name: (
                None
                if result[variant]["equal_fold_macro"][name] is None
                or baseline[name] is None
                else result[variant]["equal_fold_macro"][name] - baseline[name]
            )
            for name in METRICS
        }
        for variant in VARIANTS
        if variant != "success_only"
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    binding = args.binding.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise AblationError("ablation output must be a new directory")
    audit = trainer.load_binding(binding, args.binding_sha256)
    inventory = validate_complete_inventory(audit)
    output.mkdir(parents=True)
    summaries: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in VARIANTS:
        summaries[variant] = {}
        for body in trainer.BODIES:
            fold_output = output / variant / f"outer_lobo_{body}"
            command = fold_command(
                python_executable=args.python_executable,
                binding=binding,
                binding_sha256=args.binding_sha256,
                output=fold_output,
                held_out_body=body,
                variant=variant,
            )
            result = subprocess.run(command, check=False)
            if result.returncode != 0:
                raise AblationError(f"{variant}/{body} training exited {result.returncode}")
            summary_path = fold_output / "training_summary.json"
            if not summary_path.is_file():
                raise AblationError(f"{variant}/{body} did not produce a summary")
            summaries[variant][body] = json.loads(summary_path.read_text(encoding="utf-8"))
    # Deliberate two-phase boundary: no heldout NPZ is opened until every one
    # of the 20 runs has completed source-only checkpoint selection.
    device = torch.device("cuda")
    heldout_evaluations: dict[str, dict[str, Mapping[str, Any]]] = {}
    for variant in VARIANTS:
        heldout_evaluations[variant] = {}
        for body in trainer.BODIES:
            heldout_evaluations[variant][body] = evaluate_posthoc_heldout_fold(
                summaries[variant][body], audit,
                held_out_body=body, variant=variant, device=device,
            )
    document = {
        "format": FORMAT,
        "status": STATUS,
        "binding": str(binding),
        "binding_file_sha256": args.binding_sha256,
        "inventory": inventory,
        "fixed_budget": {
            "split_seed": SPLIT_SEED,
            "steps_per_member": STEPS_PER_MEMBER,
            "eval_every_steps": EVAL_EVERY,
            "batch_size_rows": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "variants": list(VARIANTS),
            "folds_per_variant": len(trainer.BODIES),
            "heldout_labels_used_for_checkpoint_selection": False,
            "all_checkpoints_selected_before_any_heldout_payload_open": True,
            "variant_selection_performed": False,
            "heldout_results_reporting_only": True,
        },
        "results": {
            "source_validation_member_mean": aggregate_variants(summaries),
            "posthoc_heldout": aggregate_posthoc_heldout(heldout_evaluations),
        },
    }
    trainer.core.atomic_json(output / "offline_ablation_summary.json", document)
    print("OFFLINE_ABLATION_COMPLETE=" + json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AblationError", "METRICS", "VARIANTS", "aggregate_posthoc_heldout",
    "aggregate_variants", "evaluate_posthoc_heldout_fold", "fold_command",
    "summarize_fold", "validate_complete_inventory",
]
