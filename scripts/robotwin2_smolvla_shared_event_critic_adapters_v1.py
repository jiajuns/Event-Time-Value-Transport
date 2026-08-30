#!/usr/bin/env python3
"""Production adapters for the RoboTwin2 SmolVLA EE16 shared event critic.

This module wires the policy-independent plugin protocol to the already frozen
RoboTwin2 five-body path.  It deliberately keeps the policy-native EE16 action
set separate from the canonical 14-D physical-effect tensor: the scorer sees
the latter, while the environment executor accepts only a typed selection from
the former.

The state observer is explicitly privileged.  It reads RoboTwin simulator
object poses and the analytic ``move_can_pot`` event specification; it is an
online simulator upper bound, not an actor-visible or real-robot observer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import collect_robotwin2_five_body_ee_candidate_branches_v1 as collector
import robotwin2_cross_body_canonical_adapter_v1 as canonical_adapter
import shared_event_critic_plugin_protocol_v1 as plugin


FORMAT = "etsf_robotwin2_smolvla_shared_event_critic_adapters_v1"
POLICY_FAMILY = "smolvla_universal_dual_ee16_five_body_actor"
NATIVE_ACTION_SCHEMA = "dual_arm_absolute_ee_xyz_quaternion_wxyz_gripper_16d_v1"
SUPPORTED_CANDIDATE_COUNTS = tuple(plugin.SUPPORTED_CANDIDATE_COUNTS)
DEFAULT_CANDIDATE_COUNT = 4
EXECUTED_PREFIX_STEPS = 5
PLANNED_DT_SECONDS = EXECUTED_PREFIX_STEPS / collector.SOURCE_EVENT_SAMPLING_HZ
FORMAL_MAX_ACTION_CALLS = collector.FORMAL_MAX_STEPS

NATIVE_ACTION_SEMANTIC_CONTRACT = {
    "format": "etsf_robotwin2_dual_arm_absolute_ee16_native_action_v1",
    "shape": ["horizon", 16],
    "channels": [
        "left_x",
        "left_y",
        "left_z",
        "left_quaternion_w",
        "left_quaternion_x",
        "left_quaternion_y",
        "left_quaternion_z",
        "left_gripper",
        "right_x",
        "right_y",
        "right_z",
        "right_quaternion_w",
        "right_quaternion_x",
        "right_quaternion_y",
        "right_quaternion_z",
        "right_gripper",
    ],
    "pose_semantics": "absolute_world_frame_tcp_xyz_quaternion_wxyz",
    "gripper_domain": [0.0, 1.0],
    "environment_call": "task.take_action(native_ee16, action_type=ee)",
    "not_the_canonical_effect_schema": True,
}
NATIVE_ACTION_SEMANTICS_EVIDENCE_SHA256 = plugin.canonical_sha256(
    NATIVE_ACTION_SEMANTIC_CONTRACT
)

QUERY_SEED_BINDING_CONTRACT = {
    "format": "etsf_robotwin2_scene_query_seed_binding_v1",
    "inputs": ["scene_seed", "query_index"],
    "encoding": "canonical_json_sha256_interpreted_as_unsigned_big_endian_integer",
    "scene_seed_domain": "non_negative_integer",
    "query_index_domain": "non_negative_integer",
    "query_seed_reuse_across_query_indices_allowed": False,
}

EFFECT_SEMANTIC_CONTRACT = {
    "format": "etsf_robotwin2_ee16_to_canonical_effect14_semantics_v1",
    "source_action_schema": NATIVE_ACTION_SCHEMA,
    "target_action_schema": plugin.CANONICAL_ACTION_SCHEMA,
    "source_root": "pre_candidate_current_dual_tcp_and_gripper_ee16",
    "conversion": "consecutive_absolute_ee16_to_world_delta_xyz_relative_axis_angle_and_gripper_delta",
    "canonical_adapter_contract": canonical_adapter.contract(),
    "candidate_order_preserved_bit_exact": True,
    "outcomes_or_post_execution_observations_used": False,
    "canonical_effect_is_never_an_environment_command": True,
}
EFFECT_SEMANTIC_CONTRACT_SHA256 = plugin.canonical_sha256(
    EFFECT_SEMANTIC_CONTRACT
)

PRIVILEGED_OBSERVER_CONTRACT = {
    "format": "etsf_robotwin2_privileged_move_can_pot_state27_observer_v1",
    "target_state_schema": plugin.CANONICAL_STATE_SCHEMA,
    "observation_authority": "privileged_robotwin_simulator_object_poses_and_tcp",
    "actor_visible": False,
    "real_robot_deployable": False,
    "event_spec_sha256": collector.EVENT_SPEC_SHA256,
    "success_at_pre_candidate_root": False,
    "event_age_clock": "monotone_counted_simulator_seconds",
    "wall_clock_used": False,
}

EXECUTION_BASE_CONTRACT = {
    "format": "etsf_robotwin2_native_ee16_environment_execution_v1",
    "native_action_schema": NATIVE_ACTION_SCHEMA,
    "executed_prefix_steps": EXECUTED_PREFIX_STEPS,
    "max_policy_action_calls": FORMAL_MAX_ACTION_CALLS,
    "environment_call": "task.take_action(native_ee16, action_type=ee)",
    "canonical_effect_execution_forbidden": True,
    "ordinary_planner_failure": "native_task_semantics",
    "python_exception": "propagate_as_protocol_failure",
    "terminal_before_five": "stop_without_padding_actions",
}


class Robotwin2SmolVLAAdapterError(RuntimeError):
    """A runtime input violated the frozen SmolVLA EE16 adapter contract."""


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Robotwin2SmolVLAAdapterError(f"{name} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise Robotwin2SmolVLAAdapterError(
            f"{name} must be a lowercase SHA-256"
        ) from error
    if value != value.lower():
        raise Robotwin2SmolVLAAdapterError(f"{name} must be a lowercase SHA-256")
    return value


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Robotwin2SmolVLAAdapterError(f"{name} must be a non-negative integer")
    return int(value)


def _candidate_count(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value not in SUPPORTED_CANDIDATE_COUNTS
    ):
        raise Robotwin2SmolVLAAdapterError(
            "candidate count must be authority-supported 4, 8 or 16"
        )
    return int(value)


def candidate_ids(candidate_count: int) -> tuple[str, ...]:
    count = _candidate_count(candidate_count)
    return ("actor_baseline",) + tuple(
        f"flow_noise_candidate_{index:03d}" for index in range(1, count)
    )


def candidate_noise_contract(candidate_count: int) -> dict[str, Any]:
    """Reproduce the collector's antithetic sampler for an authority-bound N."""

    count = _candidate_count(candidate_count)
    indices = list(range(count))
    return {
        "distribution": "antithetic_standard_normal_pairs_each_marginal_N_0_I",
        "candidate_indices": indices,
        "base_noise_indices": [index - index % 2 for index in indices],
        "signs": [1 if index % 2 == 0 else -1 for index in indices],
        "seed_formula": (
            "(20260903 + scene_seed*1000003 + query_index*10007 + "
            "base_noise_index*101) mod (2**63-1)"
        ),
        "candidate_zero_legacy_noise_unchanged": True,
        "collector_file_sha256": file_sha256(
            _module_file(collector, "collector")
        ),
    }


def file_sha256(path: Path | str) -> str:
    """Hash one real, non-symlink file for reproducible implementation binding."""

    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        raise Robotwin2SmolVLAAdapterError(
            f"provenance source must be a real non-symlink file: {candidate}"
        )
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_file(module: Any, name: str) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str):
        raise Robotwin2SmolVLAAdapterError(f"{name} module has no source file")
    return Path(value)


def component_implementation_evidence(role: str) -> dict[str, Any]:
    """Return the exact source-file evidence used for one component digest."""

    dependency_by_role = {
        "candidate_provider": {"collector": _module_file(collector, "collector")},
        "effect_adapter": {
            "collector": _module_file(collector, "collector"),
            "canonical_adapter": _module_file(canonical_adapter, "canonical adapter"),
        },
        "privileged_state_observer": {
            "collector": _module_file(collector, "collector"),
            "analytic_event_spec": _module_file(
                collector.analytic_event, "analytic event spec"
            ),
        },
        "environment_executor": {"collector": _module_file(collector, "collector")},
    }
    if role not in dependency_by_role:
        raise Robotwin2SmolVLAAdapterError(f"unknown component role: {role}")
    paths = {"adapter_module": Path(__file__), **dependency_by_role[role]}
    return {
        "format": "etsf_shared_event_critic_component_implementation_evidence_v1",
        "adapter_format": FORMAT,
        "role": role,
        "source_file_sha256": {
            name: file_sha256(path) for name, path in sorted(paths.items())
        },
    }


def component_implementation_sha256(role: str) -> str:
    return plugin.canonical_sha256(component_implementation_evidence(role))


def bind_query_seed(scene_seed: int, query_index: int) -> int:
    """Content-bind both root seed and query index into the protocol query seed."""

    scene = _non_negative_integer(scene_seed, "scene_seed")
    query = _non_negative_integer(query_index, "query_index")
    digest = plugin.canonical_sha256(
        {
            **QUERY_SEED_BINDING_CONTRACT,
            "scene_seed": scene,
            "query_index": query,
        }
    )
    return int(digest, 16)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise Robotwin2SmolVLAAdapterError(
        f"execution task args contain a non-JSON value: {type(value).__name__}"
    )


def _ordered_native_sha256(actions: np.ndarray, candidate_ids: Sequence[str]) -> str:
    array = np.ascontiguousarray(actions, dtype=np.dtype("<f4"))
    metadata = {
        "format": "etsf_ordered_native_candidate_set_sha256_v1",
        "native_action_schema": NATIVE_ACTION_SCHEMA,
        "dtype": "float32",
        "shape": list(array.shape),
        "candidate_ids": list(candidate_ids),
        "byte_order": "little",
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _immutable_float32(value: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float32)).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SmolVLAEE16QueryObservation:
    """Identity of one pre-candidate actor query in one live RoboTwin task."""

    task: Any
    scene_seed: int
    query_index: int

    def __post_init__(self) -> None:
        if self.task is None:
            raise Robotwin2SmolVLAAdapterError("query observation requires a live task")
        _non_negative_integer(self.scene_seed, "scene_seed")
        _non_negative_integer(self.query_index, "query_index")

    @property
    def query_seed(self) -> int:
        return bind_query_seed(self.scene_seed, self.query_index)

    @property
    def task_runtime_id(self) -> int:
        return id(self.task)


@dataclass(frozen=True)
class NativeCandidateSelection:
    """One native EE16 candidate selected from an immutable ordered set."""

    action: np.ndarray
    candidate_count: int
    candidate_index: int
    candidate_id: str
    ordered_native_candidate_set_sha256: str
    native_action_schema: str
    scene_seed: int
    query_index: int
    query_seed: int
    task_runtime_id: int

    def __post_init__(self) -> None:
        action = _immutable_float32(self.action)
        if (
            action.ndim != 2
            or action.shape[0] < EXECUTED_PREFIX_STEPS
            or action.shape[1] != collector.NATIVE_EE_DIM
            or not np.isfinite(action).all()
        ):
            raise Robotwin2SmolVLAAdapterError(
                "native selection must be finite [H>=5,16]"
            )
        count = _candidate_count(self.candidate_count)
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or not 0 <= self.candidate_index < count
        ):
            raise Robotwin2SmolVLAAdapterError("candidate selection index is invalid")
        if self.candidate_id != candidate_ids(count)[self.candidate_index]:
            raise Robotwin2SmolVLAAdapterError("candidate selection id is invalid")
        _require_sha256(
            self.ordered_native_candidate_set_sha256,
            "ordered_native_candidate_set_sha256",
        )
        if self.native_action_schema != NATIVE_ACTION_SCHEMA:
            raise Robotwin2SmolVLAAdapterError("selection is not native SmolVLA EE16")
        _non_negative_integer(self.scene_seed, "scene_seed")
        _non_negative_integer(self.query_index, "query_index")
        if self.query_seed != bind_query_seed(self.scene_seed, self.query_index):
            raise Robotwin2SmolVLAAdapterError("selection query seed is not bound")
        object.__setattr__(self, "action", action)


@dataclass(frozen=True)
class OrderedNativeCandidateSet:
    """The exact ordered actor proposal shared by baseline, critic and executor."""

    actions: np.ndarray
    root_ee_action16: np.ndarray
    candidate_ids: tuple[str, ...]
    scene_seed: int
    query_index: int
    query_seed: int
    task_runtime_id: int
    native_action_schema: str = NATIVE_ACTION_SCHEMA

    def __post_init__(self) -> None:
        actions = _immutable_float32(self.actions)
        root = _immutable_float32(self.root_ee_action16)
        if (
            actions.ndim != 3
            or actions.shape[0] not in SUPPORTED_CANDIDATE_COUNTS
            or actions.shape[1] < EXECUTED_PREFIX_STEPS
            or actions.shape[2] != collector.NATIVE_EE_DIM
            or not np.isfinite(actions).all()
        ):
            raise Robotwin2SmolVLAAdapterError(
                "ordered native candidates must be finite [N,H>=5,16], N in 4/8/16"
            )
        if root.shape != (collector.NATIVE_EE_DIM,) or not np.isfinite(root).all():
            raise Robotwin2SmolVLAAdapterError("root EE action must be finite [16]")
        if self.candidate_ids != candidate_ids(int(actions.shape[0])):
            raise Robotwin2SmolVLAAdapterError("ordered candidate ids changed")
        if self.native_action_schema != NATIVE_ACTION_SCHEMA:
            raise Robotwin2SmolVLAAdapterError("native candidate schema changed")
        _non_negative_integer(self.scene_seed, "scene_seed")
        _non_negative_integer(self.query_index, "query_index")
        if self.query_seed != bind_query_seed(self.scene_seed, self.query_index):
            raise Robotwin2SmolVLAAdapterError(
                "candidate query_seed does not bind scene_seed and query_index"
            )
        if (
            isinstance(self.task_runtime_id, bool)
            or not isinstance(self.task_runtime_id, int)
            or self.task_runtime_id <= 0
        ):
            raise Robotwin2SmolVLAAdapterError("candidate task runtime id is invalid")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "root_ee_action16", root)

    @property
    def ordered_native_candidate_set_sha256(self) -> str:
        return _ordered_native_sha256(self.actions, self.candidate_ids)

    @property
    def candidate_count(self) -> int:
        return int(self.actions.shape[0])

    def select(self, candidate_index: int) -> NativeCandidateSelection:
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise Robotwin2SmolVLAAdapterError("candidate index must be an integer")
        if not 0 <= candidate_index < self.candidate_count:
            raise Robotwin2SmolVLAAdapterError("candidate index is outside the ordered set")
        return NativeCandidateSelection(
            action=self.actions[candidate_index],
            candidate_count=self.candidate_count,
            candidate_index=candidate_index,
            candidate_id=self.candidate_ids[candidate_index],
            ordered_native_candidate_set_sha256=(
                self.ordered_native_candidate_set_sha256
            ),
            native_action_schema=self.native_action_schema,
            scene_seed=self.scene_seed,
            query_index=self.query_index,
            query_seed=self.query_seed,
            task_runtime_id=self.task_runtime_id,
        )


class SmolVLAEE16CandidateProvider:
    """Frozen SmolVLA EE16 provider backed by the formal collector sampler."""

    policy_family = POLICY_FAMILY
    native_action_schema = NATIVE_ACTION_SCHEMA

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        device: torch.device | str,
        actor_checkpoint_sha256: str,
        actor_runtime_contract_sha256: str,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        expected_instruction: str = collector.DEFAULT_INSTRUCTION,
    ) -> None:
        if policy is None or not callable(preprocessor) or not callable(postprocessor):
            raise Robotwin2SmolVLAAdapterError(
                "provider requires a loaded policy and callable processors"
            )
        if not isinstance(expected_instruction, str) or not expected_instruction.strip():
            raise Robotwin2SmolVLAAdapterError("expected instruction is invalid")
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = torch.device(device)
        self._actor_checkpoint_sha256 = _require_sha256(
            actor_checkpoint_sha256, "actor_checkpoint_sha256"
        )
        self.actor_runtime_contract_sha256 = _require_sha256(
            actor_runtime_contract_sha256, "actor_runtime_contract_sha256"
        )
        self.candidate_count = _candidate_count(candidate_count)
        self.expected_instruction = expected_instruction
        self._implementation_sha256 = component_implementation_sha256(
            "candidate_provider"
        )
        self.sampling_contract = MappingProxyType(
            {
                "format": "etsf_robotwin2_smolvla_ee16_candidate_sampling_adapter_v1",
                "actor_checkpoint_sha256": self._actor_checkpoint_sha256,
                "actor_runtime_contract_sha256": self.actor_runtime_contract_sha256,
                "provider_implementation_sha256": self._implementation_sha256,
                "collector_file_sha256": file_sha256(
                    _module_file(collector, "collector")
                ),
                "candidate_count": self.candidate_count,
                "candidate_indices": list(range(self.candidate_count)),
                "candidate_zero_is_actor_baseline": True,
                "candidate_zero_semantics": "legacy_fixed_flow_noise_actor_candidate_zero",
                "same_ordered_candidate_set_for_baseline_and_critic": True,
                "candidate_noise_contract": candidate_noise_contract(
                    self.candidate_count
                ),
                "query_seed_binding_contract": QUERY_SEED_BINDING_CONTRACT,
                "collector_sampling_inputs": ["scene_seed", "query_index"],
                "instruction": self.expected_instruction,
            }
        )
        self._sampling_contract_sha256 = plugin.canonical_sha256(
            dict(self.sampling_contract)
        )

    @property
    def actor_checkpoint_sha256(self) -> str:
        return self._actor_checkpoint_sha256

    @property
    def native_action_semantics_evidence_sha256(self) -> str:
        return NATIVE_ACTION_SEMANTICS_EVIDENCE_SHA256

    @property
    def implementation_sha256(self) -> str:
        return self._implementation_sha256

    @property
    def sampling_contract_sha256(self) -> str:
        return self._sampling_contract_sha256

    def propose_candidates(
        self,
        observation: Any,
        instruction: str,
        *,
        query_seed: int,
        candidate_count: int,
    ) -> OrderedNativeCandidateSet:
        if not isinstance(observation, SmolVLAEE16QueryObservation):
            raise Robotwin2SmolVLAAdapterError(
                "provider expects SmolVLAEE16QueryObservation"
            )
        if instruction != self.expected_instruction:
            raise Robotwin2SmolVLAAdapterError("actor instruction changed")
        requested_count = _candidate_count(candidate_count)
        if requested_count != self.candidate_count:
            raise Robotwin2SmolVLAAdapterError(
                "requested candidate count differs from provider authority binding"
            )
        if query_seed != observation.query_seed:
            raise Robotwin2SmolVLAAdapterError(
                "query_seed does not bind this scene_seed and query_index"
            )
        root_before = collector.current_ee_action16(observation.task)
        candidates = collector.generate_candidates(
            policy=self.policy,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            task=observation.task,
            instruction=instruction,
            scene_seed=observation.scene_seed,
            query_index=observation.query_index,
            candidate_count=requested_count,
            device=self.device,
        )
        root_after = collector.current_ee_action16(observation.task)
        if not np.array_equal(root_before, root_after):
            raise Robotwin2SmolVLAAdapterError(
                "candidate generation changed the live RoboTwin root"
            )
        return OrderedNativeCandidateSet(
            actions=candidates,
            root_ee_action16=root_before,
            candidate_ids=candidate_ids(self.candidate_count),
            scene_seed=observation.scene_seed,
            query_index=observation.query_index,
            query_seed=query_seed,
            task_runtime_id=observation.task_runtime_id,
        )


class SmolVLAEE16CanonicalEffectAdapter:
    """Convert absolute EE16 proposals to canonical effect14 without outcomes."""

    source_action_schema = NATIVE_ACTION_SCHEMA
    target_action_schema = plugin.CANONICAL_ACTION_SCHEMA

    def __init__(self) -> None:
        self._implementation_sha256 = component_implementation_sha256(
            "effect_adapter"
        )

    @property
    def implementation_sha256(self) -> str:
        return self._implementation_sha256

    @property
    def semantic_contract_sha256(self) -> str:
        return EFFECT_SEMANTIC_CONTRACT_SHA256

    def adapt_candidates(
        self,
        root_observation: Any,
        native_candidates: Any,
        canonical_state: plugin.CanonicalStateObservation,
        authority: plugin.AuthorityProvenance,
    ) -> plugin.CanonicalCandidateBatch:
        if not isinstance(root_observation, SmolVLAEE16QueryObservation):
            raise Robotwin2SmolVLAAdapterError(
                "effect adapter expects SmolVLAEE16QueryObservation"
            )
        if not isinstance(native_candidates, OrderedNativeCandidateSet):
            raise Robotwin2SmolVLAAdapterError(
                "effect adapter expects an ordered native EE16 candidate set"
            )
        if not isinstance(canonical_state, plugin.CanonicalStateObservation):
            raise Robotwin2SmolVLAAdapterError(
                "effect adapter expects CanonicalStateObservation"
            )
        if not isinstance(authority, plugin.AuthorityProvenance):
            raise Robotwin2SmolVLAAdapterError("effect adapter authority is invalid")
        if (
            root_observation.task_runtime_id != native_candidates.task_runtime_id
            or root_observation.scene_seed != native_candidates.scene_seed
            or root_observation.query_index != native_candidates.query_index
            or root_observation.query_seed != native_candidates.query_seed
        ):
            raise Robotwin2SmolVLAAdapterError(
                "root observation differs from the candidate proposal root"
            )
        if (
            authority.native_action_schema != self.source_action_schema
            or authority.canonical_action_schema != self.target_action_schema
            or authority.effect_adapter_implementation_sha256
            != self.implementation_sha256
            or authority.effect_adapter_semantic_contract_sha256
            != self.semantic_contract_sha256
            or authority.candidate_count != native_candidates.candidate_count
            or canonical_state.observer_implementation_sha256
            != authority.state_observer_implementation_sha256
        ):
            raise Robotwin2SmolVLAAdapterError(
                "effect adapter runtime differs from authority"
            )
        live_root = collector.current_ee_action16(root_observation.task)
        if not np.array_equal(live_root, native_candidates.root_ee_action16):
            raise Robotwin2SmolVLAAdapterError(
                "live root changed after native candidates were proposed"
            )
        effects = np.stack(
            [
                collector.canonical_action_chunk(
                    native_candidates.root_ee_action16, candidate
                )
                for candidate in native_candidates.actions
            ]
        ).astype(np.float32)
        device = canonical_state.state.device
        dtype = canonical_state.state.dtype
        horizon = int(effects.shape[1])
        count = native_candidates.candidate_count
        state = canonical_state.state[None].expand(count, -1).clone()
        actions = torch.as_tensor(effects, device=device, dtype=dtype)
        step = torch.arange(horizon, device=device)
        action_mask = (step[None] < EXECUTED_PREFIX_STEPS).expand(
            count, -1
        ).clone()
        return plugin.CanonicalCandidateBatch(
            state=state,
            actions=actions,
            action_mask=action_mask,
            action_available=torch.ones(
                count, dtype=torch.bool, device=device
            ),
            action_schema_id=torch.zeros(
                count, dtype=torch.long, device=device
            ),
            body_id=torch.zeros(count, dtype=torch.long, device=device),
            dt=torch.full(
                (count,),
                float(canonical_state.planned_dt_seconds),
                dtype=dtype,
                device=device,
            ),
            current_event_id=torch.full(
                (count,),
                int(canonical_state.current_event_id),
                dtype=torch.long,
                device=device,
            ),
            event_age_seconds=torch.full(
                (count,),
                float(canonical_state.event_age_seconds),
                dtype=dtype,
                device=device,
            ),
            remaining_action_budget=torch.full(
                (count,),
                float(canonical_state.remaining_action_budget),
                dtype=dtype,
                device=device,
            ),
            candidate_ids=native_candidates.candidate_ids,
            baseline_candidate_index=0,
            canonical_state_schema=plugin.CANONICAL_STATE_SCHEMA,
            canonical_action_schema=plugin.CANONICAL_ACTION_SCHEMA,
            authority_logical_sha256=authority.logical_sha256,
            ordered_native_candidate_set_sha256=(
                native_candidates.ordered_native_candidate_set_sha256
            ),
        )


@dataclass(frozen=True)
class PrivilegedRoboTwin2StateHistory:
    """Simulator object-pose trajectory and its counted physical timestamps."""

    trajectory: np.ndarray
    sim_times: np.ndarray
    object_names: tuple[str, ...]
    task_runtime_id: int
    scene_seed: int
    query_index: int

    def __post_init__(self) -> None:
        trajectory = _immutable_float32(self.trajectory)
        sim_times = np.asarray(self.sim_times, dtype=np.float64).copy()
        sim_times.setflags(write=False)
        if (
            trajectory.ndim != 3
            or trajectory.shape[0] < 1
            or trajectory.shape[2] != 7
            or trajectory.shape[1] != len(self.object_names)
            or not np.isfinite(trajectory).all()
        ):
            raise Robotwin2SmolVLAAdapterError(
                "privileged history trajectory must be finite [T,N,7]"
            )
        if (
            sim_times.shape != (trajectory.shape[0],)
            or not np.isfinite(sim_times).all()
            or (len(sim_times) > 1 and np.any(np.diff(sim_times) <= 0.0))
        ):
            raise Robotwin2SmolVLAAdapterError(
                "privileged history requires aligned increasing simulator time"
            )
        if (
            not self.object_names
            or len(set(self.object_names)) != len(self.object_names)
            or any(not isinstance(name, str) or not name for name in self.object_names)
        ):
            raise Robotwin2SmolVLAAdapterError("privileged object names are invalid")
        if (
            isinstance(self.task_runtime_id, bool)
            or not isinstance(self.task_runtime_id, int)
            or self.task_runtime_id <= 0
        ):
            raise Robotwin2SmolVLAAdapterError(
                "privileged history task runtime id is invalid"
            )
        _non_negative_integer(self.scene_seed, "scene_seed")
        _non_negative_integer(self.query_index, "query_index")
        object.__setattr__(self, "trajectory", trajectory)
        object.__setattr__(self, "sim_times", sim_times)


@dataclass(frozen=True)
class PrivilegedRoboTwin2TaskContext:
    calibration: Mapping[str, Any]
    remaining_action_budget: int
    planned_dt_seconds: float = PLANNED_DT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, Mapping):
            raise Robotwin2SmolVLAAdapterError("task calibration must be a mapping")
        remaining = self.remaining_action_budget
        if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining <= 0:
            raise Robotwin2SmolVLAAdapterError(
                "remaining action budget must be a positive integer"
            )
        if not math.isclose(
            float(self.planned_dt_seconds),
            PLANNED_DT_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Robotwin2SmolVLAAdapterError(
                "planned dt must remain the formal five-action 15 Hz chunk"
            )
        object.__setattr__(
            self, "calibration", MappingProxyType(dict(self.calibration))
        )


class PrivilegedRoboTwin2State27Observer:
    """Privileged RoboTwin simulator state27 observer; not actor-visible."""

    target_state_schema = plugin.CANONICAL_STATE_SCHEMA
    task_event_contract_sha256 = collector.EVENT_SPEC_SHA256
    actor_visible = False
    real_robot_deployable = False

    def __init__(self, *, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self._implementation_sha256 = component_implementation_sha256(
            "privileged_state_observer"
        )

    @property
    def implementation_sha256(self) -> str:
        return self._implementation_sha256

    def observe_state(
        self,
        observation: Any,
        history: Any,
        task_context: Any,
    ) -> plugin.CanonicalStateObservation:
        if not isinstance(observation, SmolVLAEE16QueryObservation):
            raise Robotwin2SmolVLAAdapterError(
                "privileged observer expects SmolVLAEE16QueryObservation"
            )
        if not isinstance(history, PrivilegedRoboTwin2StateHistory):
            raise Robotwin2SmolVLAAdapterError(
                "privileged observer expects simulator pose/time history"
            )
        if not isinstance(task_context, PrivilegedRoboTwin2TaskContext):
            raise Robotwin2SmolVLAAdapterError(
                "privileged observer expects PrivilegedRoboTwin2TaskContext"
            )
        if (
            history.task_runtime_id != observation.task_runtime_id
            or history.scene_seed != observation.scene_seed
            or history.query_index != observation.query_index
        ):
            raise Robotwin2SmolVLAAdapterError(
                "privileged history belongs to a different task root or query"
            )
        live_sim_time = collector._sim_time(observation.task)
        if not math.isclose(
            live_sim_time,
            float(history.sim_times[-1]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Robotwin2SmolVLAAdapterError(
                "privileged history is stale relative to the live simulator clock"
            )
        names = list(history.object_names)
        moving_name = str(task_context.calibration.get("moving", ""))
        if moving_name not in names:
            raise Robotwin2SmolVLAAdapterError(
                "privileged history lacks the calibrated moving object"
            )
        predicates, events = collector.derive_predicates_and_events(
            history.trajectory,
            history.sim_times,
            names,
            False,
            task_context.calibration,
        )
        current_event = int(events[-1])
        moving_index = names.index(moving_name)
        current_ee = collector.current_ee_action16(observation.task)
        state = collector._state27(
            poses=history.trajectory,
            names=names,
            step=len(history.trajectory) - 1,
            initial_moving_position=history.trajectory[0, moving_index, :3],
            ee_action=current_ee,
            event=current_event,
            predicates=predicates,
            calibration=task_context.calibration,
        )
        event_age = collector.event_age_seconds(events, history.sim_times)
        return plugin.CanonicalStateObservation(
            state=torch.as_tensor(state, dtype=torch.float32, device=self.device),
            current_event_id=current_event,
            event_age_seconds=float(event_age),
            remaining_action_budget=float(task_context.remaining_action_budget),
            planned_dt_seconds=float(task_context.planned_dt_seconds),
            state_schema=self.target_state_schema,
            observer_implementation_sha256=self.implementation_sha256,
        )


@dataclass(frozen=True)
class NativeEEExecutionReceipt:
    requested_seed: int
    query_index: int
    candidate_index: int
    candidate_id: str
    ordered_native_candidate_set_sha256: str
    executed_native_prefix_sha256: str
    executed_action_count: int
    episode_done: bool


class Robotwin2NativeEEEnvironmentExecutor:
    """Reset RoboTwin and execute only typed native EE16 candidate selections."""

    native_action_schema = NATIVE_ACTION_SCHEMA

    def __init__(
        self,
        *,
        task_class: Any,
        task_args: Mapping[str, Any],
        instruction: str = collector.DEFAULT_INSTRUCTION,
    ) -> None:
        if not inspect.isclass(task_class):
            raise Robotwin2SmolVLAAdapterError("task_class must be a class")
        if not isinstance(task_args, Mapping):
            raise Robotwin2SmolVLAAdapterError("task_args must be a mapping")
        if not isinstance(instruction, str) or not instruction.strip():
            raise Robotwin2SmolVLAAdapterError("executor instruction is invalid")
        self.task_class = task_class
        self.task_args = dict(task_args)
        self.task_args["step_lim"] = FORMAL_MAX_ACTION_CALLS
        self.instruction = instruction
        self._task: Any | None = None
        self._requested_seed: int | None = None
        self._query_index = 0
        self._implementation_sha256 = component_implementation_sha256(
            "environment_executor"
        )
        source_path = inspect.getsourcefile(task_class)
        task_source_sha = (
            file_sha256(Path(source_path))
            if isinstance(source_path, str) and Path(source_path).is_file()
            else None
        )
        execution_contract = {
            **EXECUTION_BASE_CONTRACT,
            "executor_implementation_sha256": self._implementation_sha256,
            "task_class_module": str(task_class.__module__),
            "task_class_qualname": str(task_class.__qualname__),
            "task_class_source_file_sha256": task_source_sha,
            "task_args_logical_sha256": plugin.canonical_sha256(
                _plain_json(self.task_args)
            ),
            "instruction": self.instruction,
        }
        self.execution_contract = MappingProxyType(execution_contract)
        self._execution_contract_sha256 = plugin.canonical_sha256(
            dict(execution_contract)
        )

    @property
    def implementation_sha256(self) -> str:
        return self._implementation_sha256

    @property
    def execution_contract_sha256(self) -> str:
        return self._execution_contract_sha256

    @property
    def task(self) -> Any:
        if self._task is None:
            raise Robotwin2SmolVLAAdapterError("executor has not been reset")
        return self._task

    def reset(self, requested_seed: int) -> SmolVLAEE16QueryObservation:
        seed = _non_negative_integer(requested_seed, "requested_seed")
        self.close()
        self._task = collector._new_task(
            self.task_class, self.task_args, seed, self.instruction
        )
        self._requested_seed = seed
        self._query_index = 0
        return self.query_observation()

    def query_observation(self) -> SmolVLAEE16QueryObservation:
        if self._task is None or self._requested_seed is None:
            raise Robotwin2SmolVLAAdapterError("executor has not been reset")
        return SmolVLAEE16QueryObservation(
            task=self._task,
            scene_seed=self._requested_seed,
            query_index=self._query_index,
        )

    def execute_candidate(
        self,
        native_candidate: Any,
        *,
        executed_prefix_steps: int,
    ) -> NativeEEExecutionReceipt:
        if not isinstance(native_candidate, NativeCandidateSelection):
            raise Robotwin2SmolVLAAdapterError(
                "executor accepts only a selection from the native EE16 candidate set"
            )
        if (
            isinstance(executed_prefix_steps, bool)
            or executed_prefix_steps != EXECUTED_PREFIX_STEPS
        ):
            raise Robotwin2SmolVLAAdapterError(
                "formal native execution is fixed to five actions"
            )
        task = self.task
        if (
            native_candidate.task_runtime_id != id(task)
            or native_candidate.scene_seed != self._requested_seed
            or native_candidate.query_index != self._query_index
            or native_candidate.query_seed
            != bind_query_seed(int(self._requested_seed), self._query_index)
        ):
            raise Robotwin2SmolVLAAdapterError(
                "native selection belongs to a different task root or query"
            )
        if collector._episode_done(task, FORMAL_MAX_ACTION_CALLS):
            raise Robotwin2SmolVLAAdapterError(
                "cannot execute a candidate after episode termination"
            )
        executed = []
        for action in native_candidate.action[:EXECUTED_PREFIX_STEPS]:
            if collector._episode_done(task, FORMAL_MAX_ACTION_CALLS):
                break
            native_action = np.asarray(action, dtype=np.float32).copy()
            if native_action.shape != (collector.NATIVE_EE_DIM,):
                raise Robotwin2SmolVLAAdapterError(
                    "executor received a non-native action row"
                )
            task.take_action(native_action, action_type="ee")
            executed.append(native_action)
        if not executed:
            raise Robotwin2SmolVLAAdapterError("candidate executed no native actions")
        prefix = np.stack(executed).astype(np.float32)
        prefix_sha = hashlib.sha256(prefix.tobytes(order="C")).hexdigest()
        receipt = NativeEEExecutionReceipt(
            requested_seed=int(self._requested_seed),
            query_index=self._query_index,
            candidate_index=native_candidate.candidate_index,
            candidate_id=native_candidate.candidate_id,
            ordered_native_candidate_set_sha256=(
                native_candidate.ordered_native_candidate_set_sha256
            ),
            executed_native_prefix_sha256=prefix_sha,
            executed_action_count=len(executed),
            episode_done=collector._episode_done(task, FORMAL_MAX_ACTION_CALLS),
        )
        self._query_index += 1
        return receipt

    def close(self) -> None:
        task = self._task
        self._task = None
        self._requested_seed = None
        self._query_index = 0
        if task is not None:
            close = getattr(task, "close_env", None)
            if callable(close):
                close(clear_cache=False)

    def __enter__(self) -> "Robotwin2NativeEEEnvironmentExecutor":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def materialize_authority(
    *,
    candidate_provider: SmolVLAEE16CandidateProvider,
    effect_adapter: SmolVLAEE16CanonicalEffectAdapter,
    state_observer: PrivilegedRoboTwin2State27Observer,
    environment_executor: Robotwin2NativeEEEnvironmentExecutor,
    critic_member_checkpoint_sha256: Sequence[str],
) -> plugin.AuthorityProvenance:
    """Create and validate the exact runtime authority for these four adapters."""

    members = tuple(
        _require_sha256(value, f"critic_member_checkpoint_sha256[{index}]")
        for index, value in enumerate(critic_member_checkpoint_sha256)
    )
    authority = plugin.AuthorityProvenance(
        policy_family=candidate_provider.policy_family,
        native_action_schema=candidate_provider.native_action_schema,
        native_action_semantics_evidence_sha256=(
            candidate_provider.native_action_semantics_evidence_sha256
        ),
        actor_checkpoint_sha256=candidate_provider.actor_checkpoint_sha256,
        candidate_provider_implementation_sha256=(
            candidate_provider.implementation_sha256
        ),
        candidate_sampling_contract_sha256=(
            candidate_provider.sampling_contract_sha256
        ),
        effect_adapter_source_action_schema=effect_adapter.source_action_schema,
        effect_adapter_implementation_sha256=effect_adapter.implementation_sha256,
        effect_adapter_semantic_contract_sha256=(
            effect_adapter.semantic_contract_sha256
        ),
        state_observer_implementation_sha256=state_observer.implementation_sha256,
        environment_executor_implementation_sha256=(
            environment_executor.implementation_sha256
        ),
        environment_execution_contract_sha256=(
            environment_executor.execution_contract_sha256
        ),
        task_event_contract_sha256=state_observer.task_event_contract_sha256,
        critic_member_checkpoint_sha256=members,
        candidate_count=candidate_provider.candidate_count,
        executed_prefix_steps=EXECUTED_PREFIX_STEPS,
        candidate_zero_is_actor_baseline=True,
        same_ordered_candidate_set_for_baseline_and_critic=True,
    )
    plugin.validate_plugin_components(
        authority,
        candidate_provider=candidate_provider,
        effect_adapter=effect_adapter,
        state_observer=state_observer,
        environment_executor=environment_executor,
    )
    return authority


__all__ = [
    "DEFAULT_CANDIDATE_COUNT",
    "EFFECT_SEMANTIC_CONTRACT",
    "EFFECT_SEMANTIC_CONTRACT_SHA256",
    "EXECUTED_PREFIX_STEPS",
    "FORMAT",
    "NATIVE_ACTION_SCHEMA",
    "NATIVE_ACTION_SEMANTIC_CONTRACT",
    "NATIVE_ACTION_SEMANTICS_EVIDENCE_SHA256",
    "NativeCandidateSelection",
    "NativeEEExecutionReceipt",
    "OrderedNativeCandidateSet",
    "PLANNED_DT_SECONDS",
    "POLICY_FAMILY",
    "PRIVILEGED_OBSERVER_CONTRACT",
    "PrivilegedRoboTwin2State27Observer",
    "PrivilegedRoboTwin2StateHistory",
    "PrivilegedRoboTwin2TaskContext",
    "QUERY_SEED_BINDING_CONTRACT",
    "Robotwin2NativeEEEnvironmentExecutor",
    "Robotwin2SmolVLAAdapterError",
    "SmolVLAEE16CandidateProvider",
    "SmolVLAEE16CanonicalEffectAdapter",
    "SmolVLAEE16QueryObservation",
    "SUPPORTED_CANDIDATE_COUNTS",
    "bind_query_seed",
    "candidate_ids",
    "candidate_noise_contract",
    "component_implementation_evidence",
    "component_implementation_sha256",
    "file_sha256",
    "materialize_authority",
]
