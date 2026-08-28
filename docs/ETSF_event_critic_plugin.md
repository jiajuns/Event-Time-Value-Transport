# ETSF ensemble event critic 插件

## 定位

`openvla_etsf_event_critic_plugin.py` 是冻结 actor 外部的候选动作 critic。它加载至少
两个（推荐三个）独立随机种子的事件世界模型 checkpoint，聚合事件、成功率、持续时间、
对象变化和 future-latent 分布，并以 ensemble 分歧估计 epistemic uncertainty。插件不修改
OpenVLA 权重，也不要求在同一进程中加载训练数据。

policy latent/action 边界现由
`docs/ETSF_policy_feature_action_bridge_contract.md` 的 content-addressed contract 验证。
SmolVLA 与 OpenVLA 只共享 canonical event/action-effect interface；state hook SHA、维度、
action slots、policy row 和 adapter 实现必须分别绑定。严格高层入口不接受缺失 bridge contract
的 checkpoint，也禁止将 SmolVLA 960-D checkpoint 直接用于 OpenVLA。即使调用者关闭普通
calibration/registration guard，plugin 也不会允许没有已验证 runtime receipt 的 actor override；
旧 checkpoint 只能 monitor。非 structured 模型只声明 event-ID-only，不再声称 reversible
predicate interface。SmolVLA 在线路径使用一体化 `rerank_smolvla_candidates`，由执行 rerank
的同一个 plugin 重验 receipt。

跨策略或跨本体不是改一个字符串即可完成：

- 新策略需要独立的 `StateAdapter` 与 `PolicyAdapter`：前者对齐观测/hidden，后者将原生
  action chunk 映射到训练时动作契约，并在留出数据上校准距离和 guard；
- 新本体需要 `EmbodimentSpec` 对应的动作映射和 clock calibration；
- ensemble checkpoint 的 `contract.body_to_id/policy_to_id` 必须注册这些校准后的 id；
- 未校准或未注册的组合只允许 `predict()` 监控，默认 `select()/rerank()` 会回退 actor。

因此当前接口支持“冻结共享核心、校准小适配器后迁移”，不宣称无监督、无标定的零样本
跨本体成功率提升。

尤其不能把 SmolVLA 的 720 维 action-expert hidden 直接填充或投影成 OpenVLA 的 4096
维 hidden：两者语义和噪声条件不同。现有 SmolVLA 候选文件中的 `candidate_hidden` 还会随
候选噪声变化，并不是候选共享的观测状态。新的 schema-v5 collector 改为保存 flow denoise
前的 960-D contextualized VLM prefix state；它的 CPU schema/loader 测试已通过，但仍须先在
4090 做真实 hook smoke 和正式采集，之后训练 SmolVLA 原生 `state_input_dim=960` checkpoint。
该 checkpoint 必须冻结 `contract.state_contracts.smolvla`；在线
`StateAdapter.calibration_id` 只能使用其中的内容寻址 id。仅维度为 960 或手工填写
`calibrated=True` 不足以启用替换，插件会以 `state_contract_mismatch` 回退 actor。
仅动作维度同为 14 或 collector 已实现，都不能证明跨策略迁移。

## 结构化事件在线输入契约

`structured_events=true` 时，每次 query 必须同时提供：

- `current_event_id: [B]`：当前规范事件，整数且位于 checkpoint 事件词表内；
- `current_predicates: [B,P]`：依次为 manifest `predicate_contract.names` 中的
  `moved/lifted/near_goal/stationary/success`，数值有限且在 `[0,1]`；
- `predicate_source="derive_atomic_predicates_v1"`；
- `predicate_calibration_id=predicate_contract.event_spec_sha256`，并且只有实际使用了该
  event spec 中当前任务的 `task_calibration` 才能置 `predicate_calibrated=True`。

训练数据中的 `derive_atomic_predicates_v1` 使用 query 时刻的 simulator object poses，结合
任务标定的 moving object、anchor/goal centers、offset、`delta_move`、`delta_z`、`tau_d`、
`tau_motion` 和 `stationary_steps` 得到动态谓词；`success` 仅在终止成功时成立。在线仿真应从
同一时刻的对象位姿按同一标定重算，不能复制上一步标签。实机视觉若没有精确对象位姿，
需要另训并校准 predicate detector，再以留出集验证其误差；仿真标定本身不构成跨本体或
跨策略的零样本证据。

核心模型和插件都会对缺失、形状错误、非有限或越界谓词抛出明确错误，不再静默补全零。
插件向每个 ensemble member 显式传递
`clock_event_id=current_event_id`，使持续时间头与当前结构化事件使用同一在线锚点。谓词
来源或 event-spec hash 不匹配时仍可输出监控预测，但 guard 会以
`uncalibrated_predicate_derivation` 或 `predicate_contract_mismatch` 回退 actor。

### 非特权 state-hidden observer（独立 artifact）

审计结论是：原接口没有现成的非特权 observer。`current_event_id` 和
`current_predicates` 要么由 simulator 对象位姿真值导出，要么由调用方自己的 perception
模块提供；只保存 world-model checkpoint 并不能在实机上得到这两个输入。因此本轮正式
factual/counterfactual world-model、scoring 和 guard 均保持冻结，不因新增 observer 重训或
改变其协议。

新增的 `openvla_etsf_state_observer.py` 是一个独立的小型监督 artifact：输入已经通过
`StateAdapter` 的 query hidden（`[B,D]` 或 `[B,T,D]`），预测当前 dynamic event 和五个
predicate。训练标签来自 schema-v5 的 train/validation group：每组初始 query 加
continuation query，离线标签仍由 simulator pose 按冻结 event spec 导出；同一 logical group
不会跨 train/validation，sealed-test HDF5 只扫描 identity attrs，绝不进入 label loader。
这不是无监督学习，也不是用未来成功标签反推当前状态。

```bash
python scripts/train_openvla_etsf_state_observer.py \
  --data /remote/v5_train100 \
  --pretrained /remote/counterfactual/member_seed/best.pt \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --split-manifest /remote/counterfactual/split_manifest.json \
  --output /remote/state_observer/seed_20260831 \
  --seed 20260831 --device cuda
```

trainer 输出 `state_observer.pt`、`state_observer_manifest.json` 和
`training_summary.json`，checkpoint/manifest 镜像 config、标签来源、event-spec SHA、
train/validation logical keys、policy/body/state contracts、校准和 deployment 状态。加载时会
验证 checkpoint SHA 与所有镜像字段。trainer 无条件输出
`rerank_enabled=false / monitor_only_requires_independent_validation`；训练 validation 只能
报告 event accuracy、event NLL、predicate F1/exact match，不能授权动作替换。只有另一个不
接触 world-model sealed test 的独立 observer 校准集、显式 promotion artifact、匹配的内容
寻址 state contract，以及逐样本置信门均通过，loader 才接受 `rerank_enabled=true`。

在线不再需要对象位姿真值的接口如下；旧的显式 event/predicate `HistoryState` 路径保持兼容：

```python
from openvla_etsf_state_observer import StateHiddenEventPredicateObserver

plugin = EventCriticPlugin.from_manifest(ensemble_manifest, device="cuda")
observer = StateHiddenEventPredicateObserver.from_manifest(
    observer_manifest, device="cuda"
)
plugin.attach_state_observer(observer)

state = plugin.observe_state(
    openvla_state_adapter,
    hidden_history,
    history_mask=history_mask,
    proprio=proprio,
)
prediction = plugin.predict(state, candidates, body)  # monitor-only 允许
decision = plugin.select(prediction, candidates, body)
```

fail-closed 条件按样本执行：未校准 artifact 返回
`uncalibrated_state_observer`；artifact/world-model provenance 不一致返回
`state_observer_contract_mismatch`；通过全局校准但本样本置信度不足返回
`state_observer_confidence_below_calibrated_gate`，三者都回退 actor。observer 预测不能通过
手填 `predicate_calibrated=True` 绕过这些检查。

截至 2026-08-27，本地只完成了代码与 CPU 合成测试，4090 远端尚未训练或校准 observer；
所以当前正式远端队列仍使用冻结的 simulator-derived 结构化监督，线上 learned observer
只能视为下一阶段的 monitor-only 支线，不能宣称已提高任务成功率。

正式 counterfactual manifest 必须在 JSON 顶层和 `contract` 中一致记录：

```json
{
  "predicate_contract": {
    "names": ["moved", "lifted", "near_goal", "stationary", "success"],
    "derivation": "derive_atomic_predicates_v1",
    "source": "simulator_object_poses_at_query_step",
    "event_spec_sha256": "<64-hex>",
    "task_calibration": {"moving": "...", "tau_d": 0.0},
    "online_requires_explicit_predicates": true,
    "missing_policy": "error"
  },
  "candidate_contract": {
    "baseline_candidate_name": "deterministic",
    "fallback_index": 0
  }
}
```

聚合 checkpoint 同时冻结 `contract`、`scoring`、完整 validation-only
`scoring_selection`、temperature、normalization 和 guard；loader 逐项比较 manifest 与
checkpoint，单独修改 manifest 打分权重、选择审计或 guard 会被拒绝。`scoring_selection`
只允许预注册的 7 项小网格，guard 只允许至多 3×3 的固定分位数网格；sealed 数据不参与。

## OpenVLA 接入

```python
import torch

from example_openvla_event_critic_plugin import OpenVLAActionAdapter
from openvla_etsf_event_critic_plugin import (
    EmbodimentSpec,
    EventCriticPlugin,
    GuardConfig,
    HistoryState,
    ScoringConfig,
)

plugin = EventCriticPlugin.from_checkpoints(
    [
        "/remote/run/seed_0/event_world_model_best.pt",
        "/remote/run/seed_1/event_world_model_best.pt",
        "/remote/run/seed_2/event_world_model_best.pt",
    ],
    device="cuda",
)

# 只有完成适配/时钟校准且 checkpoint contract 已登记后才能置 True。
body = EmbodimentSpec(
    name="piper",
    body_id=0,
    beta=0.12,
    action_contract="openvla_14d",
    action_adapter_calibrated=True,
    clock_calibrated=True,
    calibration_id="piper-openvla-validation-v1",
)
adapter = OpenVLAActionAdapter(
    policy_id=0,
    calibrated=True,
    calibration_id="openvla-piper-validation-v1",
    policy_name="openvla",  # 必须与 checkpoint contract 完全一致
)

# sampled_actions: [B,C,25,14]；fallback_index 指向 actor 默认动作，而非假定为 0。
candidates = adapter.adapt(
    sampled_actions,
    body,
    fallback_index=fallback_index,
)
state = HistoryState(
    hidden=hidden_history,       # [B,T,4096]，保留 query history
    history_mask=history_mask,  # [B,T]
    current_event_id=event_id,  # [B]
    proprio=proprio,            # [B,14]
    # structured checkpoint 必填；由当前 query 的对象状态实时推导。
    current_predicates=current_predicates,  # [B,5]
    predicate_source="derive_atomic_predicates_v1",
    predicate_calibrated=True,
    predicate_calibration_id=event_spec_sha256,
)
decision = plugin.rerank(
    state,
    candidates,
    body,
    scoring=ScoringConfig(
        gamma=0.99,
        uncertainty_weight=0.1,
        distance_weight=0.05,
    ),
    guard=GuardConfig(
        minimum_score_margin=0.05,
        maximum_candidate_distance=0.25,
        maximum_total_uncertainty=0.75,
    ),
)

# [B,25,14]，保持 OpenVLA/RoboTwin 原生执行动作，不执行 padded canonical action。
action_to_execute = decision.selected_execution_actions
```

反事实微调完成后应优先加载训练端生成的 manifest，让温度缩放、打分常数和 validation
guard 成为一个不可拆分的部署契约：

```python
plugin = EventCriticPlugin.from_manifest(
    "/remote/run/ensemble_manifest.json",
    device="cuda",
)
decision = plugin.rerank(state, candidates, body)
```

该路径验证 `counterfactual_ensemble.pt` 的 SHA256，将 success logit 按 manifest 的
`temperature` 缩放，并严格使用训练端定义的
`success probability ensemble std + mean aleatoric`。只有 `guard.enabled=true` 且候选
相对 baseline 的 gain 与 uncertainty 同时过冻结阈值才允许替换；禁用 guard 时始终回退。
为避免 sealed-test 后重复调参，manifest 路径拒绝调用方再传 `ScoringConfig/GuardConfig`。

`fallback_reasons` 会逐样本记录 `score_margin_below_guard`、
`candidate_distance_above_guard`、`uncertainty_above_guard`、未校准适配器或 checkpoint
契约未注册等原因，便于在 validation 上冻结阈值后审计 sealed test。

## 推理输出

`prediction.outputs` 至少包含：

```text
next_event_probability
reach_probability / success_probability / outcome_probability
duration_selected_log_mean / duration_selected_log_scale
object_delta_mean / future_latent_mean
aleatoric_uncertainty
epistemic_event / epistemic_reach / epistemic_success
epistemic_outcome / epistemic_duration / epistemic_object
epistemic_future_latent / epistemic_uncertainty / total_uncertainty
post_predicate_probability / relative_transition_probability  # structured only
epistemic_predicate / epistemic_relative_transition            # structured only
```

打分先对每个 ensemble member 分别计算精确的 `E[gamma**D]`，再平均基础价值；不确定性和
actor 距离惩罚只施加一次。guard 阈值必须只在 validation seed 上选择，不能查看 sealed
test 后再调整。

## CPU 回归测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_openvla_etsf_state_observer.py \
  tests/test_openvla_etsf_event_critic_plugin.py \
  tests/test_openvla_etsf_event_world_model.py \
  tests/test_train_openvla_etsf_event_world_model.py
```

这些测试使用合成的小 checkpoint，不启动 OpenVLA、RoboTwin 或 GPU 训练。
