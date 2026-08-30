from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import shared_event_critic_plugin_protocol_v1 as plugin  # noqa: E402
import train_robotwin2_five_body_lobo_shared_event_head_v1 as trainer  # noqa: E402


def digest(index: int) -> str:
    return f"{index:064x}"


def authority(*, native_schema: str = "test_native_action_v1") -> plugin.AuthorityProvenance:
    return plugin.AuthorityProvenance(
        policy_family="test_policy",
        native_action_schema=native_schema,
        native_action_semantics_evidence_sha256=(
            digest(20) if native_schema == plugin.CANONICAL_ACTION_SCHEMA else None
        ),
        actor_checkpoint_sha256=digest(1),
        candidate_provider_implementation_sha256=digest(2),
        candidate_sampling_contract_sha256=digest(3),
        effect_adapter_source_action_schema=native_schema,
        effect_adapter_implementation_sha256=digest(4),
        effect_adapter_semantic_contract_sha256=digest(5),
        state_observer_implementation_sha256=digest(6),
        environment_executor_implementation_sha256=digest(7),
        environment_execution_contract_sha256=digest(8),
        task_event_contract_sha256=digest(9),
        critic_member_checkpoint_sha256=tuple(digest(10 + index) for index in range(5)),
    )


def canonical_batch(
    bound: plugin.AuthorityProvenance,
) -> plugin.CanonicalCandidateBatch:
    state = torch.arange(plugin.STATE_DIM, dtype=torch.float32)
    state[18:23] = 0.0
    state[19] = 1.0
    state[23:27] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    state = state[None].repeat(plugin.CANDIDATE_COUNT, 1)
    actions = torch.arange(
        plugin.CANDIDATE_COUNT * 7 * plugin.ACTION_DIM, dtype=torch.float32
    ).reshape(plugin.CANDIDATE_COUNT, 7, plugin.ACTION_DIM)
    mask = torch.arange(7)[None] < plugin.EXECUTED_PREFIX_STEPS
    return plugin.CanonicalCandidateBatch(
        state=state,
        actions=actions,
        action_mask=mask.expand(plugin.CANDIDATE_COUNT, -1).clone(),
        action_available=torch.ones(plugin.CANDIDATE_COUNT, dtype=torch.bool),
        action_schema_id=torch.zeros(plugin.CANDIDATE_COUNT, dtype=torch.long),
        body_id=torch.zeros(plugin.CANDIDATE_COUNT, dtype=torch.long),
        dt=torch.full((plugin.CANDIDATE_COUNT,), 1.0 / 3.0),
        current_event_id=torch.ones(plugin.CANDIDATE_COUNT, dtype=torch.long),
        event_age_seconds=torch.full((plugin.CANDIDATE_COUNT,), 0.5),
        remaining_action_budget=torch.full((plugin.CANDIDATE_COUNT,), 80.0),
        candidate_ids=("actor_baseline", "candidate_1", "candidate_2", "candidate_3"),
        baseline_candidate_index=0,
        canonical_state_schema=plugin.CANONICAL_STATE_SCHEMA,
        canonical_action_schema=plugin.CANONICAL_ACTION_SCHEMA,
        authority_logical_sha256=bound.logical_sha256,
        ordered_native_candidate_set_sha256=digest(30),
    )


class FakeProvider:
    policy_family = "test_policy"
    native_action_schema = "test_native_action_v1"
    actor_checkpoint_sha256 = digest(1)
    implementation_sha256 = digest(2)
    sampling_contract_sha256 = digest(3)

    def propose_candidates(
        self,
        observation: Any,
        instruction: str,
        *,
        query_seed: int,
        candidate_count: int,
    ) -> Any:
        return observation, instruction, query_seed, candidate_count


class FakeEffectAdapter:
    source_action_schema = "test_native_action_v1"
    target_action_schema = plugin.CANONICAL_ACTION_SCHEMA
    implementation_sha256 = digest(4)
    semantic_contract_sha256 = digest(5)

    def adapt_candidates(
        self,
        root_observation: Any,
        native_candidates: Any,
        canonical_state: plugin.CanonicalStateObservation,
        authority: plugin.AuthorityProvenance,
    ) -> plugin.CanonicalCandidateBatch:
        del root_observation, native_candidates, canonical_state
        return canonical_batch(authority)


class FakeObserver:
    target_state_schema = plugin.CANONICAL_STATE_SCHEMA
    implementation_sha256 = digest(6)
    task_event_contract_sha256 = digest(9)

    def observe_state(
        self,
        observation: Any,
        history: Any,
        task_context: Any,
    ) -> plugin.CanonicalStateObservation:
        del observation, history, task_context
        return plugin.CanonicalStateObservation(
            state=torch.tensor([0.0] * 18 + [1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 4),
            current_event_id=0,
            event_age_seconds=0.0,
            remaining_action_budget=80.0,
            planned_dt_seconds=1.0 / 3.0,
            state_schema=plugin.CANONICAL_STATE_SCHEMA,
            observer_implementation_sha256=self.implementation_sha256,
        )


class FakeExecutor:
    native_action_schema = "test_native_action_v1"
    implementation_sha256 = digest(7)
    execution_contract_sha256 = digest(8)

    def reset(self, requested_seed: int) -> Any:
        return requested_seed

    def execute_candidate(
        self,
        native_candidate: Any,
        *,
        executed_prefix_steps: int,
    ) -> Any:
        return native_candidate, executed_prefix_steps


class FixedMember(torch.nn.Module):
    def __init__(self, scores: torch.Tensor, checkpoint_sha256: str) -> None:
        super().__init__()
        self.register_buffer("scores", scores.clone())
        self.checkpoint_sha256 = checkpoint_sha256

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert batch["actions"].shape[:1] == (plugin.CANDIDATE_COUNT,)
        return {plugin.MEMBER_SCORE_KEY: self.scores.to(batch["actions"])}


def test_runtime_protocols_and_authority_binding_are_structural() -> None:
    bound = authority()
    provider = FakeProvider()
    effect = FakeEffectAdapter()
    observer = FakeObserver()
    executor = FakeExecutor()
    assert isinstance(provider, plugin.PolicyCandidateProvider)
    assert isinstance(effect, plugin.CanonicalEffectAdapter)
    assert isinstance(observer, plugin.CanonicalStateObserver)
    assert isinstance(executor, plugin.EnvironmentExecutor)
    plugin.validate_plugin_components(
        bound,
        candidate_provider=provider,
        effect_adapter=effect,
        state_observer=observer,
        environment_executor=executor,
    )
    assert bound.to_dict()["logical_sha256"] == bound.logical_sha256


def test_openvla_sized_native_action_cannot_alias_canonical_semantics() -> None:
    bound = authority(native_schema="openvla_native_action14_v1")
    provider = FakeProvider()
    provider.native_action_schema = "openvla_native_action14_v1"
    effect = FakeEffectAdapter()
    # A 14-D provider cannot bypass semantic conversion by renaming its source
    # as the canonical action-effect schema.
    effect.source_action_schema = plugin.CANONICAL_ACTION_SCHEMA
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="differ from authority"):
        plugin.validate_plugin_components(
            bound,
            candidate_provider=provider,
            effect_adapter=effect,
            state_observer=FakeObserver(),
            environment_executor=FakeExecutor(),
        )
    with pytest.raises(
        plugin.SharedEventCriticProtocolError,
        match="native_action_semantics_evidence_sha256",
    ):
        dataclasses.replace(
            authority(native_schema=plugin.CANONICAL_ACTION_SCHEMA),
            native_action_semantics_evidence_sha256=None,
        )


def test_canonical_batch_is_label_free_and_fail_closed_on_root_or_prefix_change() -> None:
    batch = canonical_batch(authority())
    assert set(batch.to_model_batch()) == {
        "state",
        "actions",
        "action_mask",
        "action_available",
        "action_schema_id",
        "body_id",
        "dt",
        "current_event_id",
        "event_age_seconds",
        "remaining_action_budget",
    }
    changed_state = batch.state.clone()
    changed_state[2, 0] += 1.0
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="bit-exact root state"):
        dataclasses.replace(batch, state=changed_state)
    changed_mask = batch.action_mask.clone()
    changed_mask[:, 5] = True
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="exactly the first five"):
        dataclasses.replace(batch, action_mask=changed_mask)


def test_scorer_is_exact_mean_minus_quarter_population_std_without_fallback() -> None:
    bound = authority()
    raw = torch.tensor(
        [
            [0.0, 0.5, 1.2, -0.5],
            [0.0, 0.6, 1.0, -0.4],
            [0.0, 0.4, 1.1, -0.3],
            [0.0, 0.7, 0.9, -0.2],
            [0.0, 0.3, 1.3, -0.1],
        ],
        dtype=torch.float32,
    )
    members = [
        FixedMember(row, digest(10 + index)).eval()
        for index, row in enumerate(raw)
    ]
    result = plugin.SharedEventCriticScorer(members, authority=bound).score(
        canonical_batch(bound)
    )
    expected_mean = raw.mean(0)
    expected_std = raw.std(0, correction=0)
    expected = expected_mean - 0.25 * expected_std
    assert torch.equal(result.member_scores, raw)
    assert torch.allclose(result.member_mean, expected_mean)
    assert torch.allclose(result.epistemic_population_std, expected_std)
    assert torch.allclose(result.risk_adjusted_scores, expected)
    assert result.selected_candidate_index == int(torch.argmax(expected)) == 2
    # There is intentionally no accepted/fallback/guard field or candidate-zero override.
    assert not hasattr(result, "accepted")
    assert not hasattr(result, "fallback_reason")


def test_scorer_rejects_wrong_ensemble_authority_and_nonfinite_member() -> None:
    bound = authority()
    finite = torch.arange(plugin.CANDIDATE_COUNT, dtype=torch.float32)
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="exactly five"):
        plugin.SharedEventCriticScorer(
            [FixedMember(finite, digest(10 + index)).eval() for index in range(4)],
            authority=bound,
        )
    members = [
        FixedMember(finite, digest(10 + index)).eval() for index in range(5)
    ]
    wrong = dataclasses.replace(canonical_batch(bound), authority_logical_sha256=digest(63))
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="authority"):
        plugin.SharedEventCriticScorer(members, authority=bound).score(wrong)
    bad_members = [
        FixedMember(finite, digest(10 + index)).eval() for index in range(5)
    ]
    bad_members[3] = FixedMember(
        torch.tensor([0.0, 1.0, float("nan"), 3.0]), digest(13)
    ).eval()
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="finite floating"):
        plugin.SharedEventCriticScorer(bad_members, authority=bound).score(
            canonical_batch(bound)
        )
    wrong_checkpoint = [
        FixedMember(finite, digest(10 + index)).eval() for index in range(5)
    ]
    wrong_checkpoint[2].checkpoint_sha256 = digest(62)
    with pytest.raises(plugin.SharedEventCriticProtocolError, match="checkpoint differs"):
        plugin.SharedEventCriticScorer(wrong_checkpoint, authority=bound)


def test_protocol_module_has_no_policy_or_environment_imports() -> None:
    path = SCRIPTS / "shared_event_critic_plugin_protocol_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "hashlib", "json", "math", "dataclasses", "typing", "torch"}


def test_current_v9_members_run_through_the_policy_independent_scorer() -> None:
    torch.manual_seed(20260831)
    bound = authority()
    members = []
    for index, checkpoint_sha256 in enumerate(
        bound.critic_member_checkpoint_sha256
    ):
        torch.manual_seed(20260831 + index)
        member = trainer.EffectAlignedSharedEventHead().eval()
        member.checkpoint_sha256 = checkpoint_sha256
        members.append(member)
    result = plugin.SharedEventCriticScorer(members, authority=bound).score(
        canonical_batch(bound)
    )
    assert trainer.MODEL_FAMILY == "terminal_consequence_utility_shared_event_head_v9"
    assert result.member_scores.shape == (
        plugin.ENSEMBLE_MEMBER_COUNT,
        plugin.CANDIDATE_COUNT,
    )
    assert torch.isfinite(result.risk_adjusted_scores).all()
    assert 0 <= result.selected_candidate_index < plugin.CANDIDATE_COUNT
