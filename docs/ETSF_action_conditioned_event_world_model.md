# 动作条件事件世界模型：设计与训练契约

## 1. 目标与边界

本模块是冻结 actor 外部的候选动作 critic，不替换或微调 OpenVLA。它学习

\[
p(e_{t+1}, D, y_{\mathrm{reach}}, y_{\mathrm{success}}, \Delta o,
z_{t+H}\mid z_t, a_{t:t+H}, b, \pi),
\]

并以带不确定性和 actor 距离约束的分数重排候选动作块。第一阶段只在已保存的
OpenVLA hidden 上训练；不同时加载 7B actor。事实 rollout 用于预训练动力学，
同状态候选分支用于后续反事实排序微调。

模块不声称把一个机器人的原始关节动作零样本发送给另一个机器人。跨本体共享的
是事件转移核心；动作坐标映射和执行时钟由本体适配器负责。

## 2. 可插拔接口

策略适配器统一输出：

```text
CandidateBatch
  actions:          float[B, N, H, A]
  action_mask:      bool[B, N, H]
  proprio:          float[B, P]
  body_id:          long[B]
  policy_id:        long[B]
  current_event_id: long[B]
  actor_distance:   float[B, N]       # 可选 guard
```

OpenVLA、ACT 和自回归策略通过多次采样产生候选；扩散策略通过不同噪声种子产生候选；
只有单一输出的策略可使用受限扰动候选。不同动作维数由策略/本体适配器补齐并提供
mask，核心模型只接收规范化的动作效果表示。

模型输出：

```text
next_event_logits
reach_logits
duration_log_mean / duration_log_scale
success_logits
object_delta_mean / object_delta_log_scale
future_semantic
score / uncertainty
```

`beta` 或 `body_id` 的速度校准只能进入 clock 分支，不能改变事件、成功率或对象变化
预测。这是跨本体因果结构与本体执行速度的硬边界。

## 3. 模型分解

1. 冻结现有 `SemanticEncoder(4096 -> 96)`，把 actor hidden 映射为 `z_t`。
2. 用 GRU 编码完整 `25 x 14` 动作块，避免 204 维人工统计量丢失动作顺序。
3. 将 proprio、current-event、body 和 policy 元信息映射到共享维度。
4. 事件核心预测 next-event、reach、success、对象变化和 future semantic。
5. 独立 clock 分支预测对数正态持续时间分布并处理右删失。
6. 部署时由 ensemble 分歧给出 epistemic uncertainty；单模型分布头给出
   aleatoric uncertainty。
7. 反事实阶段增加基线相对残差
   `r(z,a-a_actor)=MLP([Δaction_effect,z⊙Δaction_effect])-MLP(0)`；因此 actor 候选的
   残差恒为 0，场景本身的成功难度不会被动作排序支路重复学习。事实 success/event/time/
   object heads保持原结构，组内 success-changing pair/listwise/centered/actor-contrast 损失
   监督该残差。离线 OOF 与在线插件使用完全相同的 adjusted logit/entropy 路径。

候选选择分数为

\[
S(a)=p_{\mathrm{success}} + w_e\,p_{\mathrm{advance}}
+ w_v\,\mathbb E[\gamma^D]V(e_{t+1})
- \kappa U(a)-\lambda d(a,a_{\mathrm{actor}}).
\]

只有当最佳候选相对 actor 默认候选的收益超过 margin、动作距离低于上限且不确定性
低于阈值时才允许改动作，否则回退候选 0。

## 4. 事实 rollout 标签

每条 query 形成一条 transition：

```text
hidden[i]             -> current_hidden
action_chunks[i]      -> action_chunk
hidden[i+1]           -> future_hidden（末段使用 terminal_hidden）
query_steps[i]        -> start_step
min(start+25, steps)  -> post_step
event_steps           -> current_event / post_event / next-event duration
object_poses          -> post_position - pre_position
proprio               -> current_proprio
episode success       -> success
```

next event 在 episode 结束前出现时，duration 为精确观测；没有出现时，以
`steps - start_step` 作为右删失下界。训练/验证/测试必须按 episode seed 划分，不能
把同一 episode 的 query 分到不同集合。

事实 rollout 只覆盖 actor 的 on-policy 动作分布，因此该阶段 checkpoint 是动力学
预训练结果，不自动获得反事实动作排序授权。

## 5. 反事实分支 schema v5

后续候选采集在执行首个候选 chunk 后额外记录：

```text
post_chunk_hidden
post_chunk_step
pre_object_poses / post_object_poses
pre_proprio / post_proprio
raw_event_names / raw_event_steps
event_names / event_steps
post_event_id / next_event_id
duration / duration_observed
branches/*/object_poses
branches/*/proprio
branches/*/query_steps / query_post_steps
branches/*/query_hidden / query_post_hidden
branches/*/query_actions / query_action_mask
```

每个候选必须从相同 simulator seed、resolved seed 和固定 instruction 重置。候选执行后
继续使用冻结的确定性 actor，终局 success/steps 用于组内排序损失。逐步 pose/proprio
轨迹用于构造动态谓词；operational regress 要求 phase 下降持续至少 3 个 simulator
state，recovery 要求随后恢复旧 peak 持续至少 3 states 或到达 terminal success/eK，
单独谓词 down-flip 和短阈值抖动均不计入。只有首次命中事件时间无法监督 recovery。
v5 还保留 deterministic continuation 的每个策略 query，使成功分支中的
晚期 `e3/e4`、回退和恢复状态具有对应 hidden/action，可作为 auxiliary factual
transition 训练；这些 continuation 行不能重复参与首候选的组内 ranking。旧 schema v2
只能训练 success/ranking head，schema v4 只有首候选动作转移，均不能伪造缺失监督。

## 6. 训练阶段与授权门

1. `factual-pretrain`：150 条完整 rollout，训练 event/time/object/future-latent。
2. `branch-finetune`：schema v5 同状态候选分支，增加动态谓词、continuation 转移、
   baseline-relative action residual 与组内 ranking。
3. `validation-guard`：只在 validation seed 选择 margin、距离和不确定性阈值。
4. `sealed-test`：冻结 checkpoint 和 guard 后只评测一次 test seed。

至少检查：next-event macro-F1、reach AUC、duration MAE/删失 NLL、success AUC/Brier、
对象位移 NLL、future-latent cosine、组内 pair accuracy、ECE、guard 后成功率及 bootstrap
置信区间。只有 validation 上不劣于默认策略且改变过至少一个候选时，才能将
`action_ranking_authorized` 写为 true。

## 7. 跨策略与跨本体验收

- 跨策略：leave-one-policy-out；只训练新 `PolicyAdapter`，共享核心冻结。
- 跨本体：leave-one-embodiment-out；只训练 `BodyAdapter` 和 clock calibration。
- 零样本：仅在动作效果坐标、对象事件观测和控制时基已经标准化时报告。
- 所有对比保持候选数量、推理预算和动作距离 guard 相同。

当前 `move_can_pot / Piper / OpenVLA` 只有一个 body 和一个 policy，能验证可插拔接口
与事件动力学，但不能单独证明跨本体或跨算法。相应结论必须由后续多本体、多策略
数据支持。

当前 expanded development 数据固定为旧100×4候选与新增150×5候选，共250场景/1150分支；
5×50 OOF 每折200场景训练250步，保持与旧100场景实验相同的样本曝光次数。该设置只在
development 上决定是否允许一次 fresh50；在 fresh50 得到显著改善前仍不能声称成功率提升。
