#!/usr/bin/env python3
"""Pure, dependency-injected evaluation400 v4 executor integration.

This module models the WORM execution/receipt layer without launching a
simulator, subprocess, or reading target data.  Production adapters can supply
the same interfaces later while retaining these exact schemas and chronology.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping, Protocol, Sequence

import smolvla_piper_evaluation400_audit_contract_v1 as audit
import run_smolvla_piper_evaluation400_condition_v4 as runner


LEDGER_EVENT_FORMAT = "etsf_smolvla_piper_evaluation400_worm_ledger_event_v4"
LEDGER_EVENT_STATUS = "append_only_hash_chained"
PAIR_SPEC_FORMAT = "etsf_smolvla_piper_evaluation400_pair_spec_v4"
PAIR_SPEC_STATUS = "frozen_preoutcome_pair_identity"
TERMINAL_FORMAT = "etsf_smolvla_piper_evaluation400_execution_terminal_v4"
TERMINAL_STATUS = "complete_400_pairs_1600_conditions_encrypted_targets"
EVENT_TYPES = {
    "lane_started",
    "pair_started",
    "root_prediction_precommit",
    "condition_started",
    "recovery_prediction_pre_step",
    "condition_terminal",
    "pair_terminal",
    "lane_terminal",
}


class ExecutorV4Error(RuntimeError):
    """The v4 WORM/integration boundary failed closed."""


class RootPreparer(Protocol):
    def __call__(
        self, pair: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...


BackendFactory = Callable[[Mapping[str, Any], str], runner.ConditionBackendV4]


def _require_sha(value: Any, role: str) -> str:
    if not audit.is_sha256(value):
        raise ExecutorV4Error(f"{role} must be exact SHA-256")
    return str(value)


def _require_int(
    value: Any, role: str, *, expected: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise ExecutorV4Error(f"{role} must be an exact non-bool integer")
    if expected is not None and value != expected:
        raise ExecutorV4Error(f"{role} differs from exact authority")
    return value


def _signed(base: Mapping[str, Any], field: str) -> dict[str, Any]:
    normalized = dict(base)
    return {**normalized, field: audit.canonical_sha256(normalized)}


def _verify(
    value: Mapping[str, Any], *, field: str, fields: set[str], role: str,
) -> str:
    if not isinstance(value, Mapping) or set(value) != fields | {field}:
        raise ExecutorV4Error(f"{role} fields changed")
    logical = value.get(field)
    _require_sha(logical, f"{role} SHA")
    base = {key: child for key, child in value.items() if key != field}
    if logical != audit.canonical_sha256(base):
        raise ExecutorV4Error(f"{role} canonical SHA mismatch")
    return str(logical)


def build_pair_spec(
    *, pair_id: str, pair_ordinal: int, shared_snapshot_sha256: str,
) -> dict[str, Any]:
    _require_sha(pair_id, "pair ID")
    _require_sha(shared_snapshot_sha256, "shared snapshot")
    _require_int(pair_ordinal, "pair ordinal")
    base = {
        "format": PAIR_SPEC_FORMAT,
        "status": PAIR_SPEC_STATUS,
        "pair_id": pair_id,
        "pair_ordinal": pair_ordinal,
        "shared_snapshot_sha256": shared_snapshot_sha256,
        "condition_order": list(runner.CONDITION_NAMES),
        "condition_count": len(runner.CONDITION_NAMES),
        "attempt": 0,
        "retry_count": 0,
    }
    return _signed(base, "pair_identity_sha256")


def validate_pair_spec(value: Mapping[str, Any], *, expected_ordinal: int) -> str:
    fields = {
        "format", "status", "pair_id", "pair_ordinal",
        "shared_snapshot_sha256", "condition_order", "condition_count",
        "attempt", "retry_count",
    }
    logical = _verify(
        value, field="pair_identity_sha256", fields=fields, role="pair spec"
    )
    if (
        value.get("format") != PAIR_SPEC_FORMAT
        or value.get("status") != PAIR_SPEC_STATUS
        or value.get("condition_order") != list(runner.CONDITION_NAMES)
    ):
        raise ExecutorV4Error("pair condition matrix changed")
    _require_sha(value.get("pair_id"), "pair ID")
    _require_sha(value.get("shared_snapshot_sha256"), "shared snapshot")
    for field_name, expected in (
        ("pair_ordinal", expected_ordinal),
        ("condition_count", len(runner.CONDITION_NAMES)),
        ("attempt", 0),
        ("retry_count", 0),
    ):
        _require_int(value.get(field_name), field_name, expected=expected)
    return logical


def validate_ledger_event(
    value: Mapping[str, Any], *, expected_index: int,
    expected_previous_event_sha256: str,
) -> str:
    fields = {
        "format", "status", "protocol_core_v4_sha256", "event_index",
        "event_type", "previous_event_sha256", "pair_id", "pair_ordinal",
        "condition_id", "condition_position", "step_index", "artifact_sha256",
    }
    logical = _verify(
        value, field="event_sha256", fields=fields, role="WORM ledger event"
    )
    if (
        value.get("format") != LEDGER_EVENT_FORMAT
        or value.get("status") != LEDGER_EVENT_STATUS
        or value.get("event_type") not in EVENT_TYPES
        or value.get("previous_event_sha256") != expected_previous_event_sha256
    ):
        raise ExecutorV4Error("WORM ledger chain/event type changed")
    _require_sha(value.get("protocol_core_v4_sha256"), "ledger v4 core")
    _require_sha(value.get("previous_event_sha256"), "previous ledger event")
    _require_int(value.get("event_index"), "ledger event index", expected=expected_index)
    event_type = value["event_type"]
    pair_scoped = event_type not in {"lane_started", "lane_terminal"}
    condition_scoped = event_type in {
        "condition_started", "recovery_prediction_pre_step", "condition_terminal"
    }
    step_scoped = event_type == "recovery_prediction_pre_step"
    if pair_scoped:
        _require_sha(value.get("pair_id"), "ledger pair ID")
        _require_int(value.get("pair_ordinal"), "ledger pair ordinal")
    elif value.get("pair_id") is not None or value.get("pair_ordinal") is not None:
        raise ExecutorV4Error("lane ledger event unexpectedly names a pair")
    if condition_scoped:
        if value.get("condition_id") not in runner.CONDITION_NAMES:
            raise ExecutorV4Error("ledger condition is outside the v4 matrix")
        _require_int(value.get("condition_position"), "ledger condition position")
        if value["condition_position"] != runner.CONDITION_NAMES.index(
            value["condition_id"]
        ):
            raise ExecutorV4Error("ledger condition position changed")
    elif value.get("condition_id") is not None or value.get(
        "condition_position"
    ) is not None:
        raise ExecutorV4Error("non-condition ledger event names a condition")
    if step_scoped:
        _require_int(value.get("step_index"), "ledger recovery step")
        if value["step_index"] < 1:
            raise ExecutorV4Error("recovery ledger event cannot be at root e0")
    elif value.get("step_index") is not None:
        raise ExecutorV4Error("non-recovery ledger event names a step")
    artifact = value.get("artifact_sha256")
    if artifact is not None:
        _require_sha(artifact, "ledger artifact")
    return logical


class WormLedgerV4:
    """Append-only in-memory ledger with hash chaining and replay rejection."""

    def __init__(self, protocol_core_v4_sha256: str) -> None:
        self.protocol_core_v4_sha256 = _require_sha(
            protocol_core_v4_sha256, "v4 core"
        )
        self._events: list[dict[str, Any]] = []
        self._artifact_shas: set[str] = set()

    @property
    def events(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)

    @property
    def final_event_sha256(self) -> str:
        return self._events[-1]["event_sha256"] if self._events else audit.ZERO_SHA256

    def append(
        self, event_type: str, *, pair_id: str | None = None,
        pair_ordinal: int | None = None, condition_id: str | None = None,
        condition_position: int | None = None, step_index: int | None = None,
        artifact_sha256: str | None = None,
    ) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ExecutorV4Error("unknown WORM ledger event type")
        if artifact_sha256 is not None:
            _require_sha(artifact_sha256, "ledger artifact")
            if artifact_sha256 in self._artifact_shas:
                raise ExecutorV4Error("ledger artifact replayed")
        base = {
            "format": LEDGER_EVENT_FORMAT,
            "status": LEDGER_EVENT_STATUS,
            "protocol_core_v4_sha256": self.protocol_core_v4_sha256,
            "event_index": len(self._events),
            "event_type": event_type,
            "previous_event_sha256": self.final_event_sha256,
            "pair_id": pair_id,
            "pair_ordinal": pair_ordinal,
            "condition_id": condition_id,
            "condition_position": condition_position,
            "step_index": step_index,
            "artifact_sha256": artifact_sha256,
        }
        event = _signed(base, "event_sha256")
        validate_ledger_event(
            event,
            expected_index=len(self._events),
            expected_previous_event_sha256=self.final_event_sha256,
        )
        self._events.append(event)
        if artifact_sha256 is not None:
            self._artifact_shas.add(artifact_sha256)
        return copy.deepcopy(event)

    def validate(self) -> str:
        previous = audit.ZERO_SHA256
        seen_artifacts: set[str] = set()
        for index, event in enumerate(self._events):
            logical = validate_ledger_event(
                event,
                expected_index=index,
                expected_previous_event_sha256=previous,
            )
            artifact = event.get("artifact_sha256")
            if artifact is not None:
                if artifact in seen_artifacts:
                    raise ExecutorV4Error("ledger artifact replayed")
                seen_artifacts.add(artifact)
            previous = logical
        return previous


def validate_terminal(value: Mapping[str, Any]) -> str:
    fields = {
        "format", "status", "protocol_core_v4_sha256",
        "pair_identity_set_sha256", "ordered_pair_id",
        "ordered_pair_identity_sha256", "required_pair_count",
        "complete_pair_count",
        "required_condition_count", "complete_condition_count", "retry_count",
        "incomplete_count", "exclusion_count", "condition_order",
        "ledger_final_event_sha256", "ledger_event_count",
        "root_precommit_count", "recovery_pre_step_commit_count",
        "target_envelope_count", "target_envelope_set_sha256",
    }
    logical = _verify(
        value, field="terminal_sha256", fields=fields, role="v4 terminal"
    )
    if (
        value.get("format") != TERMINAL_FORMAT
        or value.get("status") != TERMINAL_STATUS
        or value.get("condition_order") != list(runner.CONDITION_NAMES)
    ):
        raise ExecutorV4Error("v4 terminal contract changed")
    for field_name in (
        "protocol_core_v4_sha256", "pair_identity_set_sha256",
        "ledger_final_event_sha256", "target_envelope_set_sha256",
    ):
        _require_sha(value.get(field_name), f"terminal {field_name}")
    ordered_pair_id = value.get("ordered_pair_id")
    ordered_pair_identity = value.get("ordered_pair_identity_sha256")
    if (
        not isinstance(ordered_pair_id, list)
        or len(ordered_pair_id) != audit.PAIR_COUNT
        or len(set(ordered_pair_id)) != audit.PAIR_COUNT
        or any(not audit.is_sha256(item) for item in ordered_pair_id)
        or not isinstance(ordered_pair_identity, list)
        or len(ordered_pair_identity) != audit.PAIR_COUNT
        or len(set(ordered_pair_identity)) != audit.PAIR_COUNT
        or any(not audit.is_sha256(item) for item in ordered_pair_identity)
        or audit.canonical_sha256(ordered_pair_identity)
        != value["pair_identity_set_sha256"]
    ):
        raise ExecutorV4Error("terminal exact pair identity inventory changed")
    for field_name, expected in (
        ("required_pair_count", audit.PAIR_COUNT),
        ("complete_pair_count", audit.PAIR_COUNT),
        ("required_condition_count", audit.CONDITION_COUNT),
        ("complete_condition_count", audit.CONDITION_COUNT),
        ("retry_count", 0), ("incomplete_count", 0), ("exclusion_count", 0),
        ("root_precommit_count", audit.PAIR_COUNT),
        ("target_envelope_count", audit.CONDITION_COUNT),
    ):
        _require_int(value.get(field_name), f"terminal {field_name}", expected=expected)
    _require_int(value.get("ledger_event_count"), "terminal ledger event count")
    _require_int(
        value.get("recovery_pre_step_commit_count"),
        "terminal recovery commitment count",
    )
    return logical


class Evaluation400ExecutorV4:
    """Orchestrate the exact 400x4 lane through injected in-memory backends."""

    def __init__(
        self, *, protocol_core_v4_sha256: str,
        schema6_runtime_contract_sha256: str,
        evaluator_public_key_raw: bytes,
        root_preparer: RootPreparer,
        backend_factory: BackendFactory,
        dense_event_targets_fn: Callable[..., Mapping[str, Any]],
        recovery_targets_fn: Callable[..., Mapping[str, Any]],
        object_target_fn: Callable[..., Mapping[str, Any]],
        causal_observer_authority: Mapping[str, Any],
    ) -> None:
        self.protocol_core_v4_sha256 = _require_sha(
            protocol_core_v4_sha256, "v4 core"
        )
        self.schema6_runtime_contract_sha256 = _require_sha(
            schema6_runtime_contract_sha256, "runtime contract"
        )
        if not isinstance(evaluator_public_key_raw, bytes) or len(
            evaluator_public_key_raw
        ) != 32:
            raise ExecutorV4Error("evaluator public key must be exact 32 bytes")
        self.evaluator_public_key_raw = evaluator_public_key_raw
        self.root_preparer = root_preparer
        self.backend_factory = backend_factory
        self.dense_event_targets_fn = dense_event_targets_fn
        self.recovery_targets_fn = recovery_targets_fn
        self.object_target_fn = object_target_fn
        self.causal_observer_authority = copy.deepcopy(causal_observer_authority)
        self.causal_observer_authority_sha256 = (
            runner.validate_causal_observer_authority(
                self.causal_observer_authority
            )
        )
        self.ledger = WormLedgerV4(self.protocol_core_v4_sha256)
        self.condition_results: list[dict[str, Any]] = []
        self.target_envelopes: list[dict[str, Any]] = []
        self._seen_pairs: set[str] = set()
        self._recovery_commit_shas: set[str] = set()
        self._terminal: dict[str, Any] | None = None
        self._completeness: dict[str, Any] | None = None

    def execute_pair(self, pair: Mapping[str, Any], *, expected_ordinal: int) -> None:
        if self._terminal is not None:
            raise ExecutorV4Error("cannot append a pair after lane terminal")
        pair_identity_sha = validate_pair_spec(pair, expected_ordinal=expected_ordinal)
        pair_id = str(pair["pair_id"])
        if pair_id in self._seen_pairs:
            raise ExecutorV4Error("pair identity replayed")
        self._seen_pairs.add(pair_id)
        self.ledger.append(
            "pair_started", pair_id=pair_id, pair_ordinal=expected_ordinal,
            artifact_sha256=pair_identity_sha,
        )
        root_precommit, decision_input = self.root_preparer(pair)
        root_sha = audit.validate_root_precommit(root_precommit)
        if (
            root_precommit.get("protocol_core_v4_sha256")
            != self.protocol_core_v4_sha256
            or root_precommit.get("pair_id") != pair_id
            or root_precommit.get("pair_ordinal") != expected_ordinal
            or root_precommit.get("shared_snapshot_sha256")
            != pair["shared_snapshot_sha256"]
            or root_precommit.get("authority", {}).get(
                "schema6_runtime_contract_sha256"
            ) != self.schema6_runtime_contract_sha256
        ):
            raise ExecutorV4Error("root precommit differs from frozen pair/runtime")
        runner.validate_root_decision_input(
            decision_input, root_precommit=root_precommit
        )
        root_event = self.ledger.append(
            "root_prediction_precommit",
            pair_id=pair_id,
            pair_ordinal=expected_ordinal,
            artifact_sha256=root_sha,
        )
        root_ack = runner.build_root_broker_ack(
            root_precommit,
            ledger_event_sha256=root_event["event_sha256"],
            ledger_event_index=root_event["event_index"],
        )
        root_ack_sha = runner.validate_root_broker_ack(
            root_ack,
            root_precommit=root_precommit,
            expected_ledger_event_sha256=root_event["event_sha256"],
        )
        pair_result_shas: list[str] = []
        for condition_position, condition_id in enumerate(runner.CONDITION_NAMES):
            request = runner.build_condition_request(
                protocol_core_v4_sha256=self.protocol_core_v4_sha256,
                pair_id=pair_id,
                pair_ordinal=expected_ordinal,
                condition_id=condition_id,
                shared_snapshot_sha256=pair["shared_snapshot_sha256"],
                root_prediction_commit_sha256=root_sha,
                root_ack_sha256=root_ack_sha,
                schema6_runtime_contract_sha256=(
                    self.schema6_runtime_contract_sha256
                ),
                causal_observer_authority_sha256=(
                    self.causal_observer_authority_sha256
                ),
            )
            self.ledger.append(
                "condition_started",
                pair_id=pair_id,
                pair_ordinal=expected_ordinal,
                condition_id=condition_id,
                condition_position=condition_position,
                artifact_sha256=request["request_sha256"],
            )
            last_recovery_step = 0
            condition_commit_shas: set[str] = set()

            def recovery_broker(commitment: Mapping[str, Any]) -> Mapping[str, Any]:
                nonlocal last_recovery_step
                commit_sha = audit.validate_recovery_pre_step_commitment(
                    commitment,
                    expected_pair_id=pair_id,
                    expected_condition_id=condition_id,
                    seen_commit_sha256=(
                        self._recovery_commit_shas | condition_commit_shas
                    ),
                )
                step_index = commitment["step_index"]
                if type(step_index) is not int or step_index <= last_recovery_step:
                    raise ExecutorV4Error("recovery broker step replay/out-of-order")
                event = self.ledger.append(
                    "recovery_prediction_pre_step",
                    pair_id=pair_id,
                    pair_ordinal=expected_ordinal,
                    condition_id=condition_id,
                    condition_position=condition_position,
                    step_index=step_index,
                    artifact_sha256=commit_sha,
                )
                condition_commit_shas.add(commit_sha)
                self._recovery_commit_shas.add(commit_sha)
                last_recovery_step = step_index
                return audit.build_broker_ack(
                    commitment, ledger_event_sha256=event["event_sha256"]
                )

            backend = self.backend_factory(pair, condition_id)
            result = runner.execute_condition_v4(
                request=request,
                backend=backend,
                root_precommit=root_precommit,
                root_ack=root_ack,
                decision_input=decision_input,
                recovery_broker=recovery_broker,
                evaluator_public_key_raw=self.evaluator_public_key_raw,
                dense_event_targets_fn=self.dense_event_targets_fn,
                recovery_targets_fn=self.recovery_targets_fn,
                object_target_fn=self.object_target_fn,
                causal_observer_authority=self.causal_observer_authority,
            )
            result_sha = runner.validate_condition_result(
                result,
                request=request,
                root_precommit=root_precommit,
                root_ack=root_ack,
                decision_input=decision_input,
            )
            self.ledger.append(
                "condition_terminal",
                pair_id=pair_id,
                pair_ordinal=expected_ordinal,
                condition_id=condition_id,
                condition_position=condition_position,
                artifact_sha256=result_sha,
            )
            pair_result_shas.append(result_sha)
            self.condition_results.append(copy.deepcopy(result))
            self.target_envelopes.append(copy.deepcopy(result["target_envelope"]))
        pair_terminal_sha = audit.canonical_sha256(
            {
                "pair_identity_sha256": pair_identity_sha,
                "root_prediction_commit_sha256": root_sha,
                "condition_result_sha256": pair_result_shas,
            }
        )
        self.ledger.append(
            "pair_terminal", pair_id=pair_id, pair_ordinal=expected_ordinal,
            artifact_sha256=pair_terminal_sha,
        )

    def execute_all(
        self, pairs: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._terminal is not None or self.ledger.events:
            raise ExecutorV4Error("evaluation400 v4 lane is one-shot")
        if not isinstance(pairs, Sequence) or len(pairs) != audit.PAIR_COUNT:
            raise ExecutorV4Error("v4 lane requires exact 400 frozen pairs")
        pair_identity_shas = [
            validate_pair_spec(pair, expected_ordinal=index)
            for index, pair in enumerate(pairs)
        ]
        pair_ids = [str(pair["pair_id"]) for pair in pairs]
        if len(set(pair_ids)) != audit.PAIR_COUNT:
            raise ExecutorV4Error("v4 pair identity set contains duplicates")
        pair_identity_set_sha256 = audit.canonical_sha256(pair_identity_shas)
        self.ledger.append(
            "lane_started", artifact_sha256=pair_identity_set_sha256
        )
        for ordinal, pair in enumerate(pairs):
            self.execute_pair(pair, expected_ordinal=ordinal)
        if (
            len(self.condition_results) != audit.CONDITION_COUNT
            or len(self.target_envelopes) != audit.CONDITION_COUNT
        ):
            raise ExecutorV4Error("v4 executor did not complete exact 1600 conditions")
        target_envelope_shas = [
            str(value["envelope_sha256"]) for value in self.target_envelopes
        ]
        if len(set(target_envelope_shas)) != audit.CONDITION_COUNT:
            raise ExecutorV4Error("target envelope replay detected")
        target_envelope_set_sha256 = audit.canonical_sha256(target_envelope_shas)
        coverage_sha = audit.canonical_sha256(
            {
                "pair_identity_set_sha256": pair_identity_set_sha256,
                "pair_count": audit.PAIR_COUNT,
                "condition_count": audit.CONDITION_COUNT,
                "target_envelope_set_sha256": target_envelope_set_sha256,
            }
        )
        self.ledger.append("lane_terminal", artifact_sha256=coverage_sha)
        ledger_final = self.ledger.validate()
        base = {
            "format": TERMINAL_FORMAT,
            "status": TERMINAL_STATUS,
            "protocol_core_v4_sha256": self.protocol_core_v4_sha256,
            "pair_identity_set_sha256": pair_identity_set_sha256,
            "ordered_pair_id": pair_ids,
            "ordered_pair_identity_sha256": pair_identity_shas,
            "required_pair_count": audit.PAIR_COUNT,
            "complete_pair_count": audit.PAIR_COUNT,
            "required_condition_count": audit.CONDITION_COUNT,
            "complete_condition_count": audit.CONDITION_COUNT,
            "retry_count": 0,
            "incomplete_count": 0,
            "exclusion_count": 0,
            "condition_order": list(runner.CONDITION_NAMES),
            "ledger_final_event_sha256": ledger_final,
            "ledger_event_count": len(self.ledger.events),
            "root_precommit_count": audit.PAIR_COUNT,
            "recovery_pre_step_commit_count": len(self._recovery_commit_shas),
            "target_envelope_count": audit.CONDITION_COUNT,
            "target_envelope_set_sha256": target_envelope_set_sha256,
        }
        self._terminal = _signed(base, "terminal_sha256")
        validate_terminal(self._terminal)
        self._completeness = audit.build_terminal_completeness(
            terminal_receipt_sha256=self._terminal["terminal_sha256"]
        )
        audit.validate_terminal_completeness(self._completeness)
        return copy.deepcopy(self._terminal), copy.deepcopy(self._completeness)


def decrypt_complete_target_envelopes(
    *, terminal: Mapping[str, Any], completeness: Mapping[str, Any],
    target_envelopes: Sequence[Mapping[str, Any]],
    evaluator_private_key_raw: bytes,
) -> list[dict[str, Any]]:
    terminal_sha = validate_terminal(terminal)
    audit.validate_terminal_completeness(completeness)
    if completeness.get("terminal_receipt_sha256") != terminal_sha:
        raise ExecutorV4Error("completeness gate is bound to another terminal")
    if len(target_envelopes) != audit.CONDITION_COUNT:
        raise ExecutorV4Error("decrypt requires exact 1600 target envelopes")
    envelope_shas: list[str] = []
    identities: set[tuple[str, str]] = set()
    decoded: list[dict[str, Any]] = []
    for envelope in target_envelopes:
        envelope_sha, _aad = audit.validate_target_envelope(envelope)
        identity = (str(envelope["pair_id"]), str(envelope["condition_id"]))
        if identity in identities or envelope["condition_id"] not in runner.CONDITION_NAMES:
            raise ExecutorV4Error("target envelope identity replay/condition drift")
        identities.add(identity)
        envelope_shas.append(envelope_sha)
    expected_identities = {
        (pair_id, condition_id)
        for pair_id in terminal["ordered_pair_id"]
        for condition_id in runner.CONDITION_NAMES
    }
    if (
        identities != expected_identities
        or audit.canonical_sha256(envelope_shas)
        != terminal["target_envelope_set_sha256"]
    ):
        raise ExecutorV4Error("target envelope set differs from terminal")
    for envelope in target_envelopes:
        decoded.append(
            audit.open_target_envelope(
                envelope,
                evaluator_private_key_raw=evaluator_private_key_raw,
                terminal_completeness=completeness,
                expected_protocol_core_v4_sha256=terminal[
                    "protocol_core_v4_sha256"
                ],
                expected_pair_id=envelope["pair_id"],
                expected_condition_id=envelope["condition_id"],
                expected_root_prediction_commit_sha256=envelope[
                    "root_prediction_commit_sha256"
                ],
                expected_schema6_runtime_contract_sha256=envelope[
                    "schema6_runtime_contract_sha256"
                ],
            )
        )
    return decoded


__all__ = [
    "EVENT_TYPES",
    "Evaluation400ExecutorV4",
    "ExecutorV4Error",
    "LEDGER_EVENT_FORMAT",
    "PAIR_SPEC_FORMAT",
    "TERMINAL_FORMAT",
    "WormLedgerV4",
    "build_pair_spec",
    "decrypt_complete_target_envelopes",
    "validate_ledger_event",
    "validate_pair_spec",
    "validate_terminal",
]
