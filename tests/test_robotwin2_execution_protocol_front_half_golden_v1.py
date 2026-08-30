from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import collect_robotwin2_five_body_ee_candidate_branches_v1 as primary  # noqa: E402
import collect_robotwin2_scripted_expert_root_actor_branches_v1 as supplement  # noqa: E402
import robotwin2_actor_execution_protocol_v1 as protocol  # noqa: E402
import watch_robotwin2_ee16_actor_to_five_body_branches_v1 as watcher  # noqa: E402


def _write_signed(path: Path, value: dict) -> str:
    document = dict(value)
    document["logical_sha256"] = primary.canonical_sha256(document)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return primary.sha256_file(path)


@pytest.mark.parametrize(
    ("stride", "query_count", "roots_per_query", "expected_supplement_steps"),
    [
        (5, 40, 5, [5, 5, 5, 5, 5]),
        (50, 4, 50, [10, 25, 50, 50, 50]),
    ],
)
def test_one_frozen_file_drives_watcher_primary_and_supplement_without_defaults(
    tmp_path: Path,
    stride: int,
    query_count: int,
    roots_per_query: int,
    expected_supplement_steps: list[int],
) -> None:
    path_root = tmp_path / f"root-execute{stride}"
    protocol_path = path_root / "frozen" / "actor-execution.json"
    protocol_value = protocol.execution_protocol(stride)
    protocol_sha = protocol.write_execution_protocol_file(
        protocol_path, protocol_value
    )
    expected_binding = protocol.execution_protocol_file_binding(
        protocol_path, protocol_sha, path_root=path_root
    )

    watcher.configure_execution_protocol(
        protocol_value,
        protocol_path=protocol_path,
        protocol_file_sha256=protocol_sha,
        run_root=path_root / "formal-run",
        path_root=path_root,
    )
    assert watcher.PATH_ROOT == path_root.resolve()
    assert watcher.require_execution_protocol_binding() == expected_binding
    assert len(watcher.ROOT_QUERIES) == query_count
    assert watcher.TARGET_PER_CONDITION_QUERY == roots_per_query
    assert watcher.EXPECTED_GROUPS_PER_BODY == 400
    assert watcher.EXPECTED_BRANCHES_PER_BODY == 1600
    assert watcher.EXPECTED_TOTAL_BRANCHES == 8000
    assert len(watcher.base_collection_jobs()) == (
        len(watcher.BODIES)
        * ((query_count + watcher.QUERY_BLOCK_SIZE - 1) // watcher.QUERY_BLOCK_SIZE)
    )

    command = watcher.collector_command(
        {"collector": path_root / "collector.py"},
        body="piper",
        conditions=watcher.CONDITIONS,
        seed_start=watcher.BASE_SEED_START,
        seed_count=roots_per_query,
        queries=watcher.ROOT_QUERIES[: watcher.QUERY_BLOCK_SIZE],
    )
    assert command[command.index("--path-root") + 1] == str(path_root.resolve())
    assert command[command.index("--action-exec-steps") + 1] == str(stride)
    assert command[command.index("--seed-count") + 1] == str(roots_per_query)

    primary_args = primary.parse_args(
        [
            "--body", "piper",
            "--actor-checkpoint", str(path_root / "actor"),
            "--vlm-metadata-path", str(path_root / "vlm"),
            "--robotwin-root", str(path_root / "robotwin"),
            "--event-spec", str(path_root / "event.json"),
            "--actor-execution-protocol", str(protocol_path),
            "--actor-execution-protocol-sha256", protocol_sha,
            "--path-root", str(path_root),
            "--output", str(path_root / "primary" / "piper"),
        ]
    )
    loaded, binding, loaded_root, requested, universe = (
        primary.bind_execution_protocol_arguments(primary_args)
    )
    assert loaded == protocol_value
    assert binding == expected_binding
    assert loaded_root == path_root.resolve()
    assert requested == universe == protocol_value["query_indices"]
    assert primary_args.seed_count == roots_per_query
    assert primary_args.action_exec_steps == stride
    assert primary_args.max_steps == 200

    supplement_args = supplement.parse_args(
        [
            "--body", "piper",
            "--actor-checkpoint", str(path_root / "actor"),
            "--actor-authority", str(path_root / "actor-authority.json"),
            "--vlm-metadata-path", str(path_root / "vlm"),
            "--robotwin-root", str(path_root / "robotwin"),
            "--event-spec", str(path_root / "event.json"),
            "--actor-execution-protocol", str(protocol_path),
            "--actor-execution-protocol-sha256", protocol_sha,
            "--path-root", str(path_root),
            "--output", str(path_root / "supplement" / "piper"),
        ]
    )
    loaded, binding, loaded_root = supplement.bind_execution_protocol_arguments(
        supplement_args
    )
    assert loaded == protocol_value
    assert binding == expected_binding
    assert loaded_root == path_root.resolve()
    assert supplement_args.action_exec_steps == stride
    plans = supplement.horizon_contract(loaded)["root_action_plans"]
    assert [plan["planned_steps"] for plan in plans] == expected_supplement_steps
    assert [sum(plan["action_mask"]) for plan in plans] == expected_supplement_steps

    # Diagnostics intentionally remain a five-token, label-free diagnostic for
    # both execution protocols; execute50 must not silently widen this prefix.
    assert primary.DIAGNOSTIC_ACTION_PREFIX_STEPS == 5
    assert "first_five_actions" in primary.BRANCH_DIAGNOSTIC_CONTRACT[
        "candidate_action_pairwise_rms"
    ]

    checkpoint_sha = "c" * 64
    authority_path = path_root / "actor-authority.json"
    authority_sha = _write_signed(
        authority_path,
        {
            "format": watcher.ACTOR_FORMAT,
            "path_root": str(path_root.resolve()),
            "task": primary.TASK,
            "state_action_frame_contract": primary.STATE_ACTION_FRAME_CONTRACT,
            "sampling_contract": {
                "actor_execution_protocol": protocol_value,
                "actor_execution_protocol_logical_sha256": protocol_value[
                    "logical_sha256"
                ],
                "actor_execution_protocol_binding": expected_binding,
                "actor_execution_protocol_file_sha256": protocol_sha,
            },
            "actors": {
                "piper": {
                    "frozen": True,
                    "optimizer_updates_allowed": False,
                    "candidate_count": 4,
                    "checkpoint_sha256": checkpoint_sha,
                }
            },
        },
    )
    _, observed_sha = supplement.load_actor_authority(
        authority_path,
        body="piper",
        actor_checkpoint_sha256=checkpoint_sha,
        execution_protocol=protocol_value,
        execution_protocol_binding=expected_binding,
    )
    assert observed_sha == authority_sha

    watcher.validate_training_binding_contract(
        {
            "format": watcher.BINDING_FORMAT,
            "state_action_frame_contract": watcher.STATE_ACTION_FRAME_CONTRACT,
            "path_root": str(path_root.resolve()),
            "actor_execution_protocol_binding": expected_binding,
        }
    )


def test_front_half_rejects_stride_override_and_protocol_outside_common_root(
    tmp_path: Path,
) -> None:
    path_root = tmp_path / "root"
    path_root.mkdir()
    protocol_path = tmp_path / "outside.json"
    protocol_sha = protocol.write_execution_protocol_file(
        protocol_path, protocol.execution_protocol(50)
    )
    with pytest.raises(protocol.ActorExecutionProtocolError, match="contained"):
        protocol.execution_protocol_file_binding(
            protocol_path, protocol_sha, path_root=path_root
        )

    inside = path_root / "execute50.json"
    inside_sha = protocol.write_execution_protocol_file(
        inside, protocol.execution_protocol(50)
    )
    args = primary.parse_args(
        [
            "--body", "piper",
            "--actor-checkpoint", str(path_root / "actor"),
            "--vlm-metadata-path", str(path_root / "vlm"),
            "--robotwin-root", str(path_root / "robotwin"),
            "--event-spec", str(path_root / "event.json"),
            "--actor-execution-protocol", str(inside),
            "--actor-execution-protocol-sha256", inside_sha,
            "--path-root", str(path_root),
            "--output", str(path_root / "primary" / "piper"),
            "--action-exec-steps", "5",
        ]
    )
    with pytest.raises(primary.BranchCollectionError, match="disagree"):
        primary.bind_execution_protocol_arguments(args)
