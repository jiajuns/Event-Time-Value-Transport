from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import robotwin2_smolvla_shared_event_critic_adapters_v1 as adapters  # noqa: E402
import shared_event_critic_plugin_protocol_v1 as plugin  # noqa: E402
import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer  # noqa: E402


def digest(index: int) -> str:
    return f"{index:064x}"


class FakePolicyConfig:
    image_features = ("observation.images.cam_high",)
    chunk_size = 7
    max_action_dim = 16


class FakePolicy:
    def __init__(self) -> None:
        self.config = FakePolicyConfig()
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(
        self, _observation: dict[str, Any], *, noise: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros_like(noise)
        result[..., 0:3] = noise[..., 0:3] * 0.01
        result[..., 3] = 1.0
        result[..., 7] = torch.clamp(0.5 + noise[..., 7] * 0.01, 0.0, 1.0)
        result[..., 8:11] = 0.2 + noise[..., 8:11] * 0.01
        result[..., 11] = 1.0
        result[..., 15] = torch.clamp(0.5 + noise[..., 15] * 0.01, 0.0, 1.0)
        return result


class FakeRobot:
    def get_left_tcp_pose(self) -> np.ndarray:
        return np.asarray([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])

    def get_right_tcp_pose(self) -> np.ndarray:
        return np.asarray([0.2, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])

    def get_left_gripper_val(self) -> float:
        return 0.5

    def get_right_gripper_val(self) -> float:
        return 0.5


class FakeScene:
    def __init__(self) -> None:
        self.steps = 0

    def get_timestep(self) -> float:
        return 0.01

    def step(self) -> None:
        self.steps += 1


class FakeTask:
    def __init__(self) -> None:
        self.robot = FakeRobot()
        self.scene = FakeScene()
        self.take_action_cnt = 0
        self.eval_success = False
        self.actions: list[np.ndarray] = []
        self.action_types: list[str] = []
        self.closed = False

    def setup_demo(self, *, now_ep_num: int, seed: int, is_test: bool, **kwargs: Any) -> None:
        self.setup = {
            "now_ep_num": now_ep_num,
            "seed": seed,
            "is_test": is_test,
            "kwargs": kwargs,
        }

    def set_instruction(self, *, instruction: str) -> None:
        self.instruction = instruction

    def get_obs(self) -> dict[str, Any]:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            }
        }

    def take_action(self, action: np.ndarray, *, action_type: str) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.action_types.append(action_type)
        self.take_action_cnt += 1
        self.scene.step()

    def close_env(self, *, clear_cache: bool) -> None:
        assert clear_cache is False
        self.closed = True


def calibration() -> dict[str, Any]:
    event = adapters.collector.analytic_event
    return {
        "moving": "can",
        "anchor": "pot",
        "required_objects": list(event.REQUIRED_OBJECTS),
        "goal_rule": dict(event.GOAL_RULE),
        "thresholds": dict(event.THRESHOLDS),
        "event_rules": dict(event.EVENT_RULES),
    }


def history(
    root: adapters.SmolVLAEE16QueryObservation,
) -> adapters.PrivilegedRoboTwin2StateHistory:
    poses = np.asarray(
        [
            [
                [0.10, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
                [0.00, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.12, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
                [0.00, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    return adapters.PrivilegedRoboTwin2StateHistory(
        trajectory=poses,
        sim_times=np.asarray([0.0, 0.1], dtype=np.float64),
        object_names=("can", "pot"),
        task_runtime_id=root.task_runtime_id,
        scene_seed=root.scene_seed,
        query_index=root.query_index,
    )


def components(candidate_count: int):
    provider = adapters.SmolVLAEE16CandidateProvider(
        policy=FakePolicy(),
        preprocessor=lambda value: value,
        postprocessor=lambda value: value,
        device="cpu",
        actor_checkpoint_sha256=digest(1),
        actor_runtime_contract_sha256=digest(2),
        candidate_count=candidate_count,
    )
    effect = adapters.SmolVLAEE16CanonicalEffectAdapter()
    observer = adapters.PrivilegedRoboTwin2State27Observer(device="cpu")
    executor = adapters.Robotwin2NativeEEEnvironmentExecutor(
        task_class=FakeTask,
        task_args={"task_name": "move_can_pot", "eval_mode": True},
    )
    authority = adapters.materialize_authority(
        candidate_provider=provider,
        effect_adapter=effect,
        state_observer=observer,
        environment_executor=executor,
        critic_member_checkpoint_sha256=tuple(digest(10 + i) for i in range(5)),
    )
    return provider, effect, observer, executor, authority


@pytest.mark.parametrize("candidate_count", [4, 8])
def test_authority_query_binding_and_component_hashes_are_reproducible(
    candidate_count: int,
) -> None:
    provider, effect, observer, executor, authority = components(candidate_count)
    provider_again, effect_again, observer_again, executor_again, authority_again = (
        components(candidate_count)
    )
    try:
        assert authority.candidate_count == candidate_count
        assert authority.logical_sha256 == authority_again.logical_sha256
        assert provider.implementation_sha256 == provider_again.implementation_sha256
        assert provider.sampling_contract_sha256 == (
            provider_again.sampling_contract_sha256
        )
        assert effect.implementation_sha256 == effect_again.implementation_sha256
        assert observer.implementation_sha256 == observer_again.implementation_sha256
        assert executor.implementation_sha256 == executor_again.implementation_sha256
        assert executor.execution_contract_sha256 == (
            executor_again.execution_contract_sha256
        )
        assert isinstance(provider, plugin.PolicyCandidateProvider)
        assert isinstance(effect, plugin.CanonicalEffectAdapter)
        assert isinstance(observer, plugin.CanonicalStateObserver)
        assert isinstance(executor, plugin.EnvironmentExecutor)
        assert observer.actor_visible is False
        assert observer.real_robot_deployable is False
        assert adapters.bind_query_seed(17, 0) != adapters.bind_query_seed(17, 1)
        assert provider.sampling_contract["candidate_noise_contract"] == (
            adapters.candidate_noise_contract(candidate_count)
        )
    finally:
        executor.close()
        executor_again.close()


@pytest.mark.parametrize("candidate_count,selected_index", [(4, 2), (8, 6)])
def test_native_candidates_become_canonical_batch_but_only_native_selection_executes(
    candidate_count: int, selected_index: int
) -> None:
    provider, effect, observer, executor, authority = components(candidate_count)
    try:
        root = executor.reset(37)
        for _ in range(10):
            executor.task.scene.step()
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="query_seed"
        ):
            provider.propose_candidates(
                root,
                adapters.collector.DEFAULT_INSTRUCTION,
                query_seed=root.query_seed + 1,
                candidate_count=candidate_count,
            )
        candidates = provider.propose_candidates(
            root,
            adapters.collector.DEFAULT_INSTRUCTION,
            query_seed=root.query_seed,
            candidate_count=candidate_count,
        )
        assert candidates.actions.shape == (candidate_count, 7, 16)
        assert candidates.candidate_ids == adapters.candidate_ids(candidate_count)
        assert candidates.candidate_ids[0] == "actor_baseline"
        assert not candidates.actions.flags.writeable

        state = observer.observe_state(
            root,
            history(root),
            adapters.PrivilegedRoboTwin2TaskContext(
                calibration=calibration(),
                remaining_action_budget=200,
            ),
        )
        batch = effect.adapt_candidates(root, candidates, state, authority)
        assert isinstance(batch, plugin.CanonicalCandidateBatch)
        assert batch.actions.shape == (candidate_count, 7, 14)
        assert batch.candidate_ids == candidates.candidate_ids
        assert batch.ordered_native_candidate_set_sha256 == (
            candidates.ordered_native_candidate_set_sha256
        )
        expected_effect = np.stack(
            [
                adapters.collector.canonical_action_chunk(
                    candidates.root_ee_action16, candidate
                )
                for candidate in candidates.actions
            ]
        )
        np.testing.assert_array_equal(batch.actions.numpy(), expected_effect)

        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError,
            match="only a selection from the native EE16",
        ):
            executor.execute_candidate(
                batch, executed_prefix_steps=adapters.EXECUTED_PREFIX_STEPS
            )
        selection = candidates.select(selected_index)
        receipt = executor.execute_candidate(
            selection, executed_prefix_steps=adapters.EXECUTED_PREFIX_STEPS
        )
        assert receipt.candidate_index == selected_index
        assert receipt.ordered_native_candidate_set_sha256 == (
            batch.ordered_native_candidate_set_sha256
        )
        assert receipt.executed_action_count == adapters.EXECUTED_PREFIX_STEPS
        assert executor.task.action_types == ["ee"] * adapters.EXECUTED_PREFIX_STEPS
        assert all(action.shape == (16,) for action in executor.task.actions)
        np.testing.assert_array_equal(
            np.stack(executor.task.actions),
            candidates.actions[selected_index, : adapters.EXECUTED_PREFIX_STEPS],
        )
        assert executor.query_observation().query_index == 1
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="different task root or query"
        ):
            executor.execute_candidate(
                selection, executed_prefix_steps=adapters.EXECUTED_PREFIX_STEPS
            )
    finally:
        executor.close()


def test_candidate_count_and_authority_tampering_fail_closed() -> None:
    provider, effect, observer, executor, _authority = components(4)
    try:
        root = executor.reset(5)
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="provider authority binding"
        ):
            provider.propose_candidates(
                root,
                adapters.collector.DEFAULT_INSTRUCTION,
                query_seed=root.query_seed,
                candidate_count=8,
            )
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="4, 8 or 16"
        ):
            provider.propose_candidates(
                root,
                adapters.collector.DEFAULT_INSTRUCTION,
                query_seed=root.query_seed,
                candidate_count=4.0,
            )
        for invalid_count in (6, 4.0, True):
            with pytest.raises(
                adapters.Robotwin2SmolVLAAdapterError, match="4, 8 or 16"
            ):
                adapters.SmolVLAEE16CandidateProvider(
                    policy=FakePolicy(),
                    preprocessor=lambda value: value,
                    postprocessor=lambda value: value,
                    device="cpu",
                    actor_checkpoint_sha256=digest(1),
                    actor_runtime_contract_sha256=digest(2),
                    candidate_count=invalid_count,
                )

        candidates = provider.propose_candidates(
            root,
            adapters.collector.DEFAULT_INSTRUCTION,
            query_seed=root.query_seed,
            candidate_count=4,
        )
        selection = candidates.select(1)
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="selection index"
        ):
            replace(selection, candidate_index=1.0)
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="selection index"
        ):
            replace(selection, candidate_index=True)
        with pytest.raises(
            adapters.Robotwin2SmolVLAAdapterError, match="selection id"
        ):
            replace(selection, candidate_id="flow_noise_candidate_003")
    finally:
        executor.close()


def test_n8_production_adapters_score_and_execute_with_real_v9_members() -> None:
    provider, effect, observer, executor, authority = components(8)
    try:
        root = executor.reset(41)
        for _ in range(10):
            executor.task.scene.step()
        candidates = provider.propose_candidates(
            root,
            adapters.collector.DEFAULT_INSTRUCTION,
            query_seed=root.query_seed,
            candidate_count=8,
        )
        state = observer.observe_state(
            root,
            history(root),
            adapters.PrivilegedRoboTwin2TaskContext(
                calibration=calibration(), remaining_action_budget=100
            ),
        )
        batch = effect.adapt_candidates(root, candidates, state, authority)
        members = []
        for index, checkpoint_sha256 in enumerate(
            authority.critic_member_checkpoint_sha256
        ):
            torch.manual_seed(20260831 + index)
            members.append(
                plugin.BoundCriticMember(
                    trainer.EffectAlignedSharedEventHead().eval(),
                    checkpoint_sha256,
                )
            )
        scores = plugin.SharedEventCriticScorer(
            members, authority=authority
        ).score(batch)
        assert scores.member_scores.shape == (5, 8)
        assert torch.isfinite(scores.risk_adjusted_scores).all()
        receipt = executor.execute_candidate(
            candidates.select(scores.selected_candidate_index),
            executed_prefix_steps=adapters.EXECUTED_PREFIX_STEPS,
        )
        assert receipt.candidate_index == scores.selected_candidate_index
        assert receipt.ordered_native_candidate_set_sha256 == (
            batch.ordered_native_candidate_set_sha256
        )
    finally:
        executor.close()
