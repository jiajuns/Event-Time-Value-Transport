#!/usr/bin/env python3
"""Fail-closed asset preflight for one-axis ETSF transfer experiments.

This validator is deliberately data-blind: it verifies actor/code artifacts,
schema/observer contracts and the source/target identity matrix, but never opens
rollouts or labels.  In particular, OpenVLA-Piper -> SmolVLA-Aloha is rejected
because both policy and embodiment change in the same cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "etsf_transfer_asset_preflight_v1"
READY = "ready_unconfounded_schema5"
ARTIFACT_FIELDS = {"path", "sha256"}
DOMAIN_FIELDS = {"policy", "embodiment", "actor_artifact"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    if set(value) != fields:
        raise ValueError(
            f"{name} fields differ: missing={sorted(fields-set(value))}, "
            f"extra={sorted(set(value)-fields)}"
        )


def _sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _artifact(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an artifact mapping")
    _exact(value, ARTIFACT_FIELDS, name)
    path = Path(str(value["path"])).expanduser()
    if not path.is_absolute() or not path.is_file() or not _sha(value["sha256"]):
        raise ValueError(f"{name} must bind an existing absolute file")
    if file_sha256(path) != value["sha256"]:
        raise ValueError(f"{name} artifact changed")
    return {"path": str(path.resolve()), "sha256": str(value["sha256"])}


def _domain(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    _exact(value, DOMAIN_FIELDS, name)
    policy = str(value["policy"])
    embodiment = str(value["embodiment"])
    if not policy or not embodiment:
        raise ValueError(f"{name} policy/embodiment must be non-empty")
    _artifact(value["actor_artifact"], f"{name}.actor_artifact")
    return policy, embodiment


def validate_preflight(value: Mapping[str, Any]) -> None:
    """Validate that a formal cell is unconfounded and executable as schema-v5."""

    _exact(
        value,
        {
            "format",
            "status",
            "study_id",
            "axis",
            "source_domain",
            "target_domain",
            "tasks",
            "contracts",
            "capabilities",
            "access",
        },
        "preflight",
    )
    if value["format"] != FORMAT or value["status"] != READY:
        raise ValueError("asset preflight is not ready")
    if not isinstance(value["study_id"], str) or not value["study_id"]:
        raise ValueError("study_id must be non-empty")
    axis = str(value["axis"])
    if axis not in ("policy", "embodiment"):
        raise ValueError("axis must be policy or embodiment")
    source_policy, source_body = _domain(value["source_domain"], "source_domain")
    target_policy, target_body = _domain(value["target_domain"], "target_domain")
    if axis == "policy":
        if source_policy == target_policy or source_body != target_body:
            raise ValueError(
                "policy transfer requires different policies on the identical embodiment"
            )
    elif source_policy != target_policy or source_body == target_body:
        raise ValueError(
            "embodiment transfer requires the identical policy on different embodiments"
        )
    tasks = value["tasks"]
    if (
        not isinstance(tasks, Sequence)
        or isinstance(tasks, (str, bytes))
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise ValueError("tasks must contain unique non-empty names")
    contracts = value["contracts"]
    if not isinstance(contracts, Mapping):
        raise ValueError("contracts must be a mapping")
    _exact(
        contracts,
        {
            "schema_version",
            "event_spec_sha256",
            "source_state_contract_sha256",
            "target_state_contract_sha256",
            "source_action_effect_contract_sha256",
            "target_action_effect_contract_sha256",
        },
        "contracts",
    )
    if int(contracts["schema_version"]) != 5:
        raise ValueError("formal transfer requires schema-v5 collectors")
    for name in set(contracts) - {"schema_version"}:
        if not _sha(contracts[name]):
            raise ValueError(f"contracts.{name} must be a SHA256")
    capabilities = value["capabilities"]
    if not isinstance(capabilities, Mapping):
        raise ValueError("capabilities must be a mapping")
    _exact(
        capabilities,
        {
            "source_schema5_collector",
            "target_schema5_collector",
            "deployable_observer",
            "privileged_pose_upper_bound_available",
        },
        "capabilities",
    )
    _artifact(capabilities["source_schema5_collector"], "source_schema5_collector")
    _artifact(capabilities["target_schema5_collector"], "target_schema5_collector")
    observer = capabilities["deployable_observer"]
    if not isinstance(observer, Mapping):
        raise ValueError("deployable_observer must be a mapping")
    _exact(observer, {"mode", "artifact"}, "deployable_observer")
    if observer["mode"] not in ("actor_hidden_observer", "rgb_observer"):
        raise ValueError("primary observer must be actor-hidden or RGB")
    _artifact(observer["artifact"], "deployable_observer.artifact")
    if capabilities["privileged_pose_upper_bound_available"] not in (True, False):
        raise ValueError("privileged upper-bound availability must be boolean")
    access = value["access"]
    if not isinstance(access, Mapping):
        raise ValueError("access must be a mapping")
    _exact(
        access,
        {"openvla_confirmation_labels_read", "gpu_job_started"},
        "access",
    )
    if (
        access["openvla_confirmation_labels_read"] is not False
        or access["gpu_job_started"] is not False
    ):
        raise ValueError("preflight must be label-blind and CPU-only")


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("preflight input must be a JSON object")
    validate_preflight(value)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, value)
    print(json.dumps({"status": READY, "study_id": value["study_id"]}))


if __name__ == "__main__":
    main()
