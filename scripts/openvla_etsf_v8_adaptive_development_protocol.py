#!/usr/bin/env python3
"""Fail-closed contract for the adaptive current-D250 v8 analysis.

The current development collection started before the v8 structured-head
protocol existed.  It is therefore dishonest to reuse the prospective
``created_before_target_labels_read`` contract.  This module preserves the
same fixed domains/statistics while permanently marking the resulting evidence
as adaptive, development-only, and incapable of authorising confirmation.
"""

from __future__ import annotations

from typing import Any, Mapping

from openvla_etsf_v8_structured_heads_protocol import (
    canonical_sha256,
    make_preregistration,
)


FORMAT = "etsf_v8_current_d250_adaptive_development_contract_v1"


def make_adaptive_development_contract(
    *,
    implementation_sha256: str,
    label_derivation_sha256: str,
    base_checkpoint_sha256: str,
    base_identity_contract_sha256: str,
) -> dict[str, Any]:
    """Create the immutable, explicitly non-prospective D250 contract."""

    prospective = make_preregistration(
        implementation_sha256=implementation_sha256,
        label_derivation_sha256=label_derivation_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        # Reuse only the immutable statistical template.  The adaptive D250
        # evidence has a checkpoint-bound 150-seed identity contract, not an
        # independently published hash of factual *training groups*.
        base_training_groups_sha256=base_identity_contract_sha256,
    )
    source_sha256 = dict(prospective["source_sha256"])
    source_sha256["base_identity_contract"] = source_sha256.pop(
        "base_training_groups"
    )
    value: dict[str, Any] = {
        "format": FORMAT,
        "development_only": True,
        "evidence_design": "adaptive_current_d250_after_collection_started",
        "temporal_provenance": {
            "protocol_created_before_collection_started": False,
            "protocol_created_before_target_labels_read": False,
            "prospective_claim_allowed": False,
        },
        "fresh50": dict(prospective["fresh50"]),
        "scope": {
            **dict(prospective["scope"]),
            "adaptive_development_only": True,
            "prospective_confirmation": False,
        },
        "source_sha256": source_sha256,
        "oof": dict(prospective["oof"]),
        "statistics": dict(prospective["statistics"]),
        "domains": dict(prospective["domains"]),
    }
    value["contract_sha256"] = canonical_sha256(value)
    return value


def validate_adaptive_development_contract(value: Mapping[str, Any]) -> None:
    if value.get("format") != FORMAT:
        raise RuntimeError("v8 adaptive development contract format mismatch")
    unsigned = dict(value)
    digest = unsigned.pop("contract_sha256", None)
    if digest != canonical_sha256(unsigned):
        raise RuntimeError("v8 adaptive development contract signature mismatch")
    sources = value.get("source_sha256")
    if not isinstance(sources, Mapping):
        raise RuntimeError("v8 adaptive development sources are missing")
    expected = make_adaptive_development_contract(
        implementation_sha256=str(sources.get("implementation", "")),
        label_derivation_sha256=str(sources.get("label_derivation", "")),
        base_checkpoint_sha256=str(sources.get("base_checkpoint", "")),
        base_identity_contract_sha256=str(
            sources.get("base_identity_contract", "")
        ),
    )
    if dict(value) != expected:
        raise RuntimeError("v8 adaptive development frozen contract mismatch")
    if value.get("temporal_provenance") != {
        "protocol_created_before_collection_started": False,
        "protocol_created_before_target_labels_read": False,
        "prospective_claim_allowed": False,
    }:
        raise RuntimeError("v8 adaptive temporal provenance was weakened")
    if value.get("fresh50") != {
        "inputs_accepted": False,
        "labels_read": False,
        "authorization_possible": False,
    }:
        raise RuntimeError("v8 adaptive development must never consume Fresh50")


__all__ = [
    "FORMAT",
    "make_adaptive_development_contract",
    "validate_adaptive_development_contract",
]
