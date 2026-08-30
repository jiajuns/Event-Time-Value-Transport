# 共享事件 critic 策略无关插件协议 v1

实现入口为 `scripts/shared_event_critic_plugin_protocol_v1.py`。它不加载策略、不生成候选、
不读取环境，也不导入 SmolVLA、OpenVLA 或 RoboTwin。现有正式 runner 未被改动。

## 边界

运行时组件由四个 `@runtime_checkable Protocol` 明确分开：

- `PolicyCandidateProvider`：从冻结 actor 产生四个有序原生候选；候选 0 必须是该 actor
  自己的默认动作。
- `CanonicalEffectAdapter`：把原生候选的真实物理语义转换为
  `dual_ee_se3_gripper_delta_14d_v2`，输出共享头所需的 canonical batch。
- `CanonicalStateObserver`：只负责产生当前的 27-D state、event、event age、剩余预算和物理
  `dt`。
- `EnvironmentExecutor`：按原生动作语义 reset 和执行前五步；它不执行 14-D canonical
  effect 张量。

`validate_plugin_components()` 会同时做结构检查，并逐项核对 actor checkpoint、provider、
sampling、effect adapter、observer、executor、执行合同及其 SHA-256。`AuthorityProvenance`
还绑定五个不同的 critic member checkpoint、canonical schema、候选数、候选 0 baseline 和
同一有序候选集合同。

维数相等不是语义相等。EE16 绝对双臂末端位姿只是一个外部 native schema；现有
EE16→SE(3) delta 转换器可以在仓库外包装为 `CanonicalEffectAdapter`，但本协议不会导入它。
OpenVLA 的 14-D native action 也不能因为维数是 14 就声明为 canonical effect：authority
要求 provider native schema 与 adapter source schema 精确相等；若 native schema 声称已经
等于 canonical schema，还必须绑定独立的 semantic-evidence SHA。对于 joint-space、单臂、
token 或未声明坐标系的动作，必须实现实际 FK/控制器/物理效果转换，无法转换时就不能使用
当前单 action-stem checkpoint。

## Canonical batch 与纯 scorer

`CanonicalCandidateBatch` 是一个完整决策，固定验证：

- state 为 bit-exact 同根的 `[4,27]`；
- action effect 为 `[4,H>=5,14]`，mask 恰好只开放前五步；
- 四候选共享 event、event age、physical `dt` 和剩余预算；
- `body_id=0`、`action_schema_id=0`，与当前共享单行 checkpoint 一致；
- 不含 success、event target、rank label 或环境 outcome。

`SharedEventCriticScorer` 只接受该 batch 和五个处于 eval 模式的 member。每个 member 的运行时
`checkpoint_sha256` 必须按顺序与 authority 绑定的五个 digest 精确相等，并返回有限的
`candidate_rank_logit: [4]`，随后严格计算：

```text
member_mean = mean(member_scores, axis=member)
epistemic_std = population_std(member_scores, axis=member)
risk_adjusted = member_mean - 0.25 * epistemic_std
selected = argmax(risk_adjusted)
```

这里没有 fallback、阈值、接受门或策略调用；candidate 0 不会因是 baseline 而被特殊选中。
部署是否执行所选候选属于外部实验协议，不属于 scorer。

## 最小调用方式

```python
from shared_event_critic_plugin_protocol_v1 import (
    SharedEventCriticScorer,
    validate_plugin_components,
)

validate_plugin_components(
    authority,
    candidate_provider=provider,
    effect_adapter=effect_adapter,
    state_observer=observer,
    environment_executor=executor,
)

native = provider.propose_candidates(
    observation, instruction, query_seed=query_seed, candidate_count=4
)
state = observer.observe_state(observation, history, task_context)
batch = effect_adapter.adapt_candidates(
    observation, native, state, authority
)
scores = SharedEventCriticScorer(five_members, authority=authority).score(batch)
selected_native_index = scores.selected_candidate_index
```

定向测试为 `tests/test_shared_event_critic_plugin_protocol_v1.py`，覆盖 runtime protocol、
provenance 绑定、OpenVLA 14-D 语义冒充拒绝、canonical shape/root/prefix 合同、五成员精确
`mean-0.25*population_std`、无 fallback 行为，以及禁止策略/环境模块 import。
