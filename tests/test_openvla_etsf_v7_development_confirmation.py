from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0, str(SCRIPTS))

import launch_openvla_etsf_v7_development_confirmation as launch  # noqa: E402
import evaluate_openvla_etsf_v7_development_confirmation as evaluator  # noqa: E402
import openvla_etsf_v7_development_confirmation as v7  # noqa: E402
from collect_openvla_etsf_event_branches import explicit_seed_registry  # noqa: E402
from openvla_etsf_structured_event_time_utility import (  # noqa: E402
    guarded_candidate_selection_numpy, structured_event_time_utility_numpy,
)


def candidates() -> dict:
    path = Path(__file__).resolve().parents[1] / "artifacts/protocol/v7_development_confirmation_seed_candidates_20260827.json"
    return json.loads(path.read_text(encoding="utf-8"))


def sources() -> dict:
    result = {}
    for name, count in (("official150", 150), ("development150", 150), ("fresh50", 50)):
        requested = list(range({"official150": 0, "development150": 1000, "fresh50": 2000}[name],
                               {"official150": 0, "development150": 1000, "fresh50": 2000}[name] + count))
        resolved = [value + 10_000 for value in requested]
        result[name] = {"path": f"/{name}.json", "sha256": "a" * 64,
            "requested_seeds": requested, "resolved_seeds": resolved,
            "identity_sets_sha256": v7.canonical_sha256({"requested": requested, "resolved": resolved})}
    return result


def seed_manifest() -> dict:
    selected = [{"seed": 100_101_000 + i, "requested_seed": 100_101_000 + i,
                 "resolved_seed": 40_000 + i} for i in range(250)]
    audit = [{"requested_seed": row["requested_seed"], "resolved_seed": row["resolved_seed"],
              "decision": "selected"} for row in selected]
    return v7.make_seed_manifest(
        selected=selected, audit=audit, sources=sources(),
        candidate={"payload": candidates(), "source": "/candidate.json",
                   "source_sha256": "d" * 64},
    )


def test_candidate_and_seed_manifest_are_label_free_and_exclude_all_prior_sets() -> None:
    assert len(v7.expand_candidates(candidates())) == 400
    manifest = seed_manifest()
    audit = v7.validate_seed_manifest(manifest, verify_files=False)
    assert len(audit["requested_seeds"]) == len(audit["resolved_seeds"]) == 250
    assert set(manifest["exclusion_sources"]) == {"official150", "development150", "fresh50"}
    assert manifest["fresh_confirmation_eligible"] is False


def test_reset_selection_rejects_requested_and_resolved_prior_overlap() -> None:
    selected, audit = v7.select_reset_unique_scenes(
        range(1000), resolver=lambda seed: 7 if seed == 0 else seed + 10_000,
        excluded={7, 10_001}, count=250,
    )
    assert audit[0]["decision"] == "resolved_identity_in_prior_exclusion"
    assert audit[1]["decision"] == "resolved_identity_in_prior_exclusion"
    assert len(selected) == 250


def test_preregistration_freezes_one_formula_margin_and_signed_gate() -> None:
    value = v7.make_preregistration(
        seed_manifest=seed_manifest(), source_contract={"pretrained_sha256": "b" * 64},
        task_calibration_sha256="c" * 64,
    )
    v7.validate_preregistration(value)
    assert value["candidate_contract"]["names"] == list(v7.DEPLOYMENT_CANDIDATE_NAMES)
    assert value["score_contract"]["gain_margin"] == 0.05
    assert value["statistics_contract"]["multiple_comparisons"] == 1
    assert value["fresh_confirmation"]["inputs_accepted_by_v7"] is False
    assert value["fresh_confirmation"]["authorization_possible"] == (
        "true_only_if_single_signed_gate_passes"
    )
    changed = json.loads(json.dumps(value)); changed["score_contract"]["gain_margin"] = 0.0
    with pytest.raises(RuntimeError, match="mismatch"):
        v7.validate_preregistration(changed)


def test_protocol_utility_delegates_exact_deployment_formula() -> None:
    reached = np.asarray([[0, 0, 0, 0, index] for index in range(4)], dtype=float)
    immediate = reached[::-1].copy(); duration = np.asarray([0.0, 0.1, 0.2, 0.3])
    deployed = structured_event_time_utility_numpy(
        reached, immediate, duration, event_values=v7.EVENT_VALUES
    )
    utility = v7.fixed_world_utility(reached, immediate, duration)
    np.testing.assert_allclose(utility, deployed["utility"])
    decision = v7.fixed_decision(utility)
    guard = guarded_candidate_selection_numpy(utility)
    assert decision["selected_index"] == int(guard["selected_index"])
    assert decision["utility_gain"] == float(guard["score_margin"])


def test_single_prospective_gate_can_sign_authorization_but_never_reads_fresh(monkeypatch) -> None:
    monkeypatch.setattr(v7, "BOOTSTRAP_SAMPLES", 1000)
    rows = []
    for i in range(250):
        # argmax utility is candidate1; 30 helpful and no harmful changes.
        labels = [0, 1, 0, 0] if i < 30 else [0, 0, 0, 0]
        rows.append({"logical_key": f"g{i}", "utility": [0.0, 1.0, 0.0, 0.0],
                     "success": labels})
    result = v7.evaluate_fixed_policy(rows)
    assert result["development_gate_pass"] is True
    assert result["fresh50_confirmation_authorized"] is True
    assert result["v7_reads_or_launches_fresh"] is False


def _launcher_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        candidates=tmp_path / "candidates.json", official150_manifest=tmp_path / "official.json",
        development150_manifest=tmp_path / "development.json", fresh50_manifest=tmp_path / "fresh.json",
        official_seed_registry=tmp_path / "registry.json", model_path=tmp_path / "model",
        rlinf_root=tmp_path / "rlinf", robotwin_root=tmp_path / "robotwin",
        robotwin_code=tmp_path / "robotwin_code", event_spec=tmp_path / "events.json",
        pretrained=tmp_path / "factual.pt", collection_output=tmp_path / "collection",
        state_root=tmp_path / "state", seed_resolver=SCRIPTS / "preregister_robotwin_v7_development_confirmation.py",
        collector=SCRIPTS / "collect_openvla_etsf_event_branches.py",
        evaluator=SCRIPTS / "evaluate_openvla_etsf_v7_development_confirmation.py",
        python_bin=Path(sys.executable), gpu_index=0, dry_run=True,
        recover_resolved_seeds=False,
    )


def test_launcher_is_serial_preregister_before_collection_and_has_no_fresh_input(tmp_path: Path) -> None:
    commands = launch.build_commands(_launcher_args(tmp_path))
    assert [row["stage"] for row in commands] == ["resolve_seeds", "preregister", "collect", "evaluate"]
    assert [row["uses_gpu"] for row in commands] == [False, False, True, False]
    collect = commands[2]["argv"]
    assert collect[collect.index("--blends") + 1 : collect.index("--temperature")] == ["0.25", "0.5", "0.75"]
    assert "--fresh-seed-manifest" not in collect
    assert "--v7-preregistration" in collect


def test_collector_registry_accepts_only_mutually_exclusive_v7_contract() -> None:
    assert explicit_seed_registry(allow_unregistered_seeds=True, fresh_seed_manifest=None,
        development_seed_manifest=None, v7_seed_manifest=Path("v7.json")) == (
            "explicit_v7_prospective_development"
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        explicit_seed_registry(allow_unregistered_seeds=True, fresh_seed_manifest=Path("fresh.json"),
            development_seed_manifest=None, v7_seed_manifest=Path("v7.json"))


def _write_recovery_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    args = _launcher_args(tmp_path)
    for path in (
        args.candidates,
        args.official150_manifest,
        args.development150_manifest,
        args.fresh50_manifest,
        args.official_seed_registry,
        args.event_spec,
        args.pretrained,
    ):
        path.write_text("{}", encoding="utf-8")
    for path in (args.model_path, args.rlinf_root, args.robotwin_root, args.robotwin_code):
        path.mkdir()
    plan = launch.preflight(args)
    args.state_root.mkdir()
    launch.atomic_json(args.state_root / "launch_plan.json", plan)
    failed = {
        **plan,
        "status": "failed_closed_no_fresh_authorization",
        "stage_results": {},
        "current_stage": "resolve_seeds",
        "fresh_confirmation_labels_read": False,
        "error_type": "RuntimeError",
        "error": "v7 stage resolve_seeds failed; see resolve_seeds.log",
    }
    launch.atomic_json(args.state_root / "launch_state.json", failed)
    seed_path = args.state_root / "v7_seed_manifest.json"
    launch.atomic_json(seed_path, {"signed": "manifest"})
    payload = "f" * 64

    def validate(value, *, verify_files):
        assert value == {"signed": "manifest"}
        assert verify_files is True
        return {
            "requested_seeds": list(range(250)),
            "resolved_seeds": list(range(1000, 1250)),
            "seed_manifest_payload_sha256": payload,
        }

    monkeypatch.setattr(launch, "validate_seed_manifest", validate)
    log = args.state_root / "logs" / "resolve_seeds.log"
    log.parent.mkdir()
    marker = {
        "output": str(seed_path.resolve()),
        "groups": 250,
        "labels_read": False,
        "payload_sha256": payload,
    }
    log.write_text(
        "resolver output\n" + launch.RESOLVED_SEED_MARKER + json.dumps(marker) +
        "\nsimulator teardown returned nonzero\n",
        encoding="utf-8",
    )
    return args, plan, failed


def test_recovery_preflight_accepts_only_valid_resolver_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _write_recovery_fixture(tmp_path, monkeypatch)
    recovery = launch.recovery_preflight(args)
    assert recovery["format"] == launch.RECOVERY_FORMAT
    assert [row["stage"] for row in recovery["commands"]] == [
        "preregister", "collect", "evaluate"
    ]
    assert recovery["recovered_seed_manifest"]["validated_with_verify_files"] is True
    assert recovery["recovered_seed_manifest"]["labels_read"] is False
    assert recovery["fresh_confirmation_inputs_accepted"] is False
    assert recovery["original_failure"]["failed_stage"] == "resolve_seeds"


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("wrong_stage", "only a failed resolve-seeds"),
        ("prereg_exists", "requires prereg/result/token/collection to be absent"),
        ("collection_exists", "requires prereg/result/token/collection to be absent"),
        ("marker_missing", "exactly one resolved-seed completion marker"),
        ("source_changed", "source file changed before recovery"),
    ],
)
def test_recovery_preflight_fails_closed_on_ambiguous_or_changed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    args, _, failed = _write_recovery_fixture(tmp_path, monkeypatch)
    if mutation == "wrong_stage":
        failed["current_stage"] = "preregister"
        launch.atomic_json(args.state_root / "launch_state.json", failed)
    elif mutation == "prereg_exists":
        (args.state_root / "v7_preregistration.json").write_text("{}", encoding="utf-8")
    elif mutation == "collection_exists":
        args.collection_output.mkdir()
    elif mutation == "marker_missing":
        (args.state_root / "logs" / "resolve_seeds.log").write_text(
            "teardown only\n", encoding="utf-8"
        )
    elif mutation == "source_changed":
        args.event_spec.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises((RuntimeError, FileExistsError), match=match):
        launch.recovery_preflight(args)


def test_recovery_revalidates_manifest_and_propagates_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _write_recovery_fixture(tmp_path, monkeypatch)

    def reject(*_args, **_kwargs):
        raise RuntimeError("signed source mismatch")

    monkeypatch.setattr(launch, "validate_seed_manifest", reject)
    with pytest.raises(RuntimeError, match="signed source mismatch"):
        launch.recovery_preflight(args)


def test_recovery_executes_no_resolver_and_preserves_original_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _write_recovery_fixture(tmp_path, monkeypatch)
    recovery = launch.recovery_preflight(args)
    original_state = args.state_root / "launch_state.json"
    original_log = args.state_root / "logs" / "resolve_seeds.log"
    state_before = launch.sha256(original_state)
    log_before = launch.sha256(original_log)
    called = []

    def fake_run(stage, _args, _env):
        called.append(stage["stage"])
        if stage["stage"] == "evaluate":
            return {"status": "complete", "fresh50_confirmation_authorized": False}
        return {"status": "complete"}

    monkeypatch.setattr(launch, "run_stage", fake_run)
    monkeypatch.setattr(launch, "require_exclusive_idle_gpu", lambda _index: {"idle": True})
    state = launch.execute_recovery(args, recovery)
    assert called == ["preregister", "collect", "evaluate"]
    assert state["status"] == "complete_independent_development_fresh_forbidden"
    assert state["fresh50_confirmation_authorized"] is False
    assert state["original_failure_provenance_preserved"] is True
    assert launch.sha256(original_state) == state_before
    assert launch.sha256(original_log) == log_before


def test_recovery_execute_rejects_provenance_change_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, failed = _write_recovery_fixture(tmp_path, monkeypatch)
    recovery = launch.recovery_preflight(args)
    failed["error"] = "changed after recovery preflight"
    launch.atomic_json(args.state_root / "launch_state.json", failed)
    with pytest.raises(RuntimeError, match="provenance changed after preflight"):
        launch.execute_recovery(args, recovery)
    assert not (args.state_root / "v7_recovery_plan.json").exists()
    assert not (args.state_root / "v7_recovery_state.json").exists()


def test_recovery_execute_rechecks_all_source_files_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _ = _write_recovery_fixture(tmp_path, monkeypatch)
    recovery = launch.recovery_preflight(args)
    args.event_spec.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance changed after preflight"):
        launch.execute_recovery(args, recovery)
    assert not (args.state_root / "v7_recovery_plan.json").exists()
    assert not (args.state_root / "v7_recovery_state.json").exists()


def test_recovery_launcher_is_content_addressed_by_preregistration() -> None:
    assert (
        SCRIPTS / "launch_openvla_etsf_v7_development_confirmation.py"
    ).resolve() in evaluator._implementation_files()
