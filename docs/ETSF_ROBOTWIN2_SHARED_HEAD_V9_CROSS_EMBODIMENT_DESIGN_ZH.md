# ETSF RoboTwin2 跨本体共享事件头 v9：模型、监督、插件与完整效果协议

> 更新日期：2026-08-31。对应模型族
> `terminal_consequence_utility_shared_event_head_v9`。本文区分三种证据：候选池 oracle、共享头离线预测、
> held-out 本体闭环成功率。前两者不能替代第三者；完整配对闭环结束前，不声称已经提高跨本体成功率。

## 1. 当前结论

共享头 v9 是冻结 actor 外部的动作条件事件后果模型。它对同一根状态的多个候选动作分别预测下一事件、
持续时间、对象变化、回退/恢复、有限时域终局事件、成功概率、目标进展与不确定性，再用受约束 utility
重排序。它不更新 OpenVLA/SmolVLA 权重，不生成动作，也不为 held-out 本体增加可训练参数。

截至 2026-08-31 00:54 CST，正式 C 数据已有 614/2000 decisions、2456/8000 branches：

- candidate-0 成功 1/614，N=4 success oracle 也为 1/614，mixed-success decision 为 0；
- 21/614 个 decision 的候选可改善分阶段进度，oracle 平均阶段增益为 `+0.00855`；
- 277/614 个全失败 decision 仍存在事件或目标进展差异，可用于 dense comparative 监督；
- `e0/e12/eK` 根终局计数为 260/353/1，recovery 有效样本为 0，说明中后期边界监督明显不足。

因此，当前已证明的是“候选动作会造成可学习的局部后果差异”，尚未证明共享头能提高二值成功率。v9
的结构和补采协议正是针对时域交互、e3/e4、恢复样本与候选覆盖不足，而不是继续加 gate 或用小测试代替
完整成功率实验。

## 2. 跨本体成立的操作性定义

五个本体为 `aloha-agilex / arx-x5 / franka / piper / ur5`。所有本体先解析到同一物理语义：

- 27-D canonical state：object→goal、左右 EE→object、对象位移、夹爪、对象四元数、当前事件 one-hot
  与四个解析式谓词；
- 14-D canonical action effect：左右 EE 的 `Δxyz + Δaxis-angle + Δgripper`；
- 五事件词表：`e0 / e12 / e3 / e4 / eK`；
- 物理时间：`event_age_seconds`、`dt` 与 `remaining_action_budget`。

严格 leave-one-body-out（LOBO）每折只用四个 source bodies 训练、选择 checkpoint 和校准，第五个 body
的 payload、标签和 supplement manifest 均 zero-open。模型只有一个共享 body row，没有机器人 ID
embedding、目标本体 adapter、目标域微调或目标域 normalization。这是“同任务、同事件本体、同 canonical
物理语义下的跨本体迁移”，不是无需 adapter 的任意机器人/任务通用性。

## 3. v9 网络

输入是当前 canonical 状态和 actor 候选动作效果：

```text
(state27, current_event, event_age, remaining_budget,
 action14[H], action_mask, physical_dt)
                    ↓
 shared semantic/action/transition trunk
                    ↓
 proper event/time/object/terminal heads
                    ↓
 bounded monotone consequence utility
```

### 3.1 Proper 世界后果头

每个候选输出：

- `post_event_logits`：执行前五个动作效果后的事件；
- `next_event_logits`：下一事件；
- `duration_selected_log_mean/log_scale`：到下一事件边界的持续时间分布；
- `object_delta_mean/log_scale`：对象 SE(3) 变化及 Student-t 不确定性；
- `regression_probability` 与 `joint_recovery_probability`；
- `terminal_event_logits`：在有限剩余预算内，继续运行同一 frozen actor 后到达的最大事件；
- `terminal_goal_progress_mean/log_scale`：有限时域终局目标距离进展及不确定性；
- `success_probability=p(terminal_event=eK)`，与终局事件分布严格一致。

事件转移、持续时间、对象效果、成功/恢复和终局后果都有独立监督，不把标量 progress 当成全部世界状态。

### 3.2 v9 的关键修复：时域条件化 FiLM 残差

旧 additive context 后接线性 terminal heads 时，同一根状态内两个候选的相对 logit 差会把共享时域项抵消，
导致 H=10/25/50/100/200 虽被输入，却不能真正改变候选相对排序。v9 改为：

\[
(\gamma,\beta)=f(\log(1+a_t),\log(1+H_t)),
\]

\[
\tilde h=(1+0.1\tanh\gamma)\odot h+0.1\tanh\beta,
\]

\[
h_H=\tilde h+0.1\tanh r(\tilde h).
\]

FiLM 和 residual 最后一层零初始化，训练起点保持旧 trunk 行为；调制幅度有界，避免短 horizon 补充数据
破坏主分布。定向测试要求候选相对预测会随 event age 和 remaining budget 改变。

### 3.3 可解释排序 utility

proper heads 被压成九个 `[0,1]` 后果特征：post stage、next-event advance rate、success、
no-unrecovered-regression、短期目标收益/风险、终局 stage、终局目标收益/风险。进入 utility 前整体
`detach`，收益只能取非负归一化权重，风险只能取非正有界权重。因此 listwise ranking 只学习如何组合
物理后果，不能把 proper heads 偷偷改写成自由 latent critic。

semantic comparative loss 允许事件/动作/transition/terminal context、terminal residual、terminal event
和 terminal goal mean 学习真实候选间差异，但只在这组 active union 上施加一次上限：comparative 梯度
范数最多为 proper 梯度范数的 0.1。duration、对象不确定性、terminal goal scale 和 utility 不在这条
梯度路径中。

五成员部署分数固定为：

\[
s_i=\operatorname{mean}_m U_{m,i}-0.25\operatorname{std}_{m,pop}U_{m,i}.
\]

直接选择 `argmax(s_i)`；没有接受阈值、fallback、授权 gate 或 candidate-0 特权。

## 4. 两条监督数据流

### 4.1 C：正式 actor-prefix 候选分支

C 是五本体 × 两条件 × 40 query × 五 seed = 2000 decisions；每根四候选，共 8000 branches。四候选
从 bit-exact 同一可恢复根状态出发，执行 actor 前五步候选后，再由同一 frozen actor continuation 到成功
或 action limit。C 提供主分布 normalization、proper/world/rank 训练、source validation、checkpoint
选择和 calibration。

### 4.2 B：source-only e3/e4 多时域补充

B 不把 expert 最终成功当 actor 标签。公开 scripted expert 只用于到达第一个 e3/e4 物理根，保存根后
expert 立即退出；四候选和 continuation 全部重新由同一 frozen actor 执行，所有标签均来自 actor 分支。

每个 body、condition、horizon slot 预注册 16 个有序 reserve seeds。horizon 固定为
`10/25/50/100/200`，每个 slot 只接受第一个同时存在且可 canonicalize 的 e3/e4 pair。选择发生在任何
actor candidate outcome 前；unstable reset、expert planner failure、缺 e3/e4 和 canonicalization failure
进入有序 reject ledger。reserve 耗尽时失败退出，不能伪装为完整数据。

最终仍精确为五本体 × 两条件 × 五 horizon × e3/e4 = 100 decisions、400 branches。每个 LOBO fold
只打开四个 source-body B manifests/payload；held-out B zero-open。B 以固定 `lambda=0.25` 只更新 proper
world heads：不进入 normalization、rank utility、source validation、checkpoint selection 或 calibration。

## 5. 可插拔边界

`shared_event_critic_plugin_protocol_v1.py` 把运行时拆为四个结构化 adapter：

1. `PolicyCandidateProvider`：冻结策略产生四个有序 native candidates，candidate 0 是 actor baseline；
2. `CanonicalEffectAdapter`：把 native action 的真实控制语义变为 14-D canonical physical effect；
3. `CanonicalStateObserver`：产生 state27、事件、事件年龄、剩余预算和物理 dt；
4. `EnvironmentExecutor`：按原策略动作语义 reset/执行，不把 canonical tensor 当控制命令。

插件 scorer 只接受严格 canonical batch 和五个 v9 members，并逐成员核对运行时 checkpoint SHA 与
authority，再执行 `mean-0.25*population_std` 后 argmax。
它不 import SmolVLA、OpenVLA 或 RoboTwin，也不调用策略/环境。OpenVLA 的 native action 即使也是 14-D，
也不能靠维数相同冒充 canonical effect；必须有坐标系、控制器和语义可审计的 effect adapter。

因此 v9 的 head/scorer 层对多种候选策略可插拔，但“直接迁移”有两个必要条件：策略能提供多个候选，且
native action 能无标签地映射到同一 canonical 物理效果。换事件本体或任务后还需要新的解析式 event/state
adapter；当前 checkpoint 不能跳过这些条件。

## 6. 与 VLAC、ProgressVLA 的区别

| 方法 | 主要预测/引导 | 策略关系 | 跨本体证据 | v9 的差异 |
|---|---|---|---|---|
| VLAC | 两帧图像+语言的 signed progress delta 与 done | 同一自回归模型兼任 critic/action，并在真实 RL 中更新 | 多机器人联合训练与新机器人/场景泛化描述 | v9 明确 action-conditioned event/time/object/recovery 分解，actor 冻结，做 held-out body LOBO critic-only 重排序 |
| ProgressVLA | action-conditioned latent visual future + normalized-time scalar progress | 对 diffusion latent action 做可微 classifier guidance，并联合微调 | 两阶段 latent-action/decoder 支持灵活本体；主实验是 CALVIN、LIBERO 和一个 ARX 平台 | v9 不要求 diffusion/可微策略，不回传策略梯度，使用显式事件与物理效果，并以五本体 held-out 配对 ΔSR 验证 |
| ETSF v9 | 下一事件、时长、成功/恢复、对象 SE(3)、终局后果与不确定性 | frozen-policy best-of-N 外接重排序 | 五折 LOBO、held-out supplement zero-open、配对闭环 | 主创新是结构化事件后果传输 + canonical effect + 策略无关插件 + 严格迁移因果协议的组合 |

VLAC 原文入口：https://arxiv.org/html/2509.15937 。ProgressVLA 原文入口：
https://arxiv.org/html/2603.27670 。这里不把 VLAC 的多机器人联合在线训练或 ProgressVLA 的 latent-action
灵活性等同于当前尚待完成的 held-out critic-only Δ成功率证据。

## 7. 完整效果实验

离线 event F1/NLL、duration MAE/NLL、object Student-t NLL、success Brier/NLL、terminal event/progress、
ranking 与 uncertainty 只解释机制。主结论来自每个 held-out body、clean/randomized、100 个预注册 seed
的同初始条件配对：

```text
baseline：冻结 actor，执行 raw candidate 0
treatment：同一冻结 actor + 对应 LOBO 五成员 v9，执行风险调整最高分候选
```

主指标为 `ΔSR = SR(actor+v9)-SR(actor)`；同时报告配对 bootstrap 95% CI、McNemar exact two-sided p、
阶段进度 `0/0.25/0.5/0.75/1.0` 的配对差，以及 body、condition 和宏平均结果。N=4 保留主可比协议，
N=8 使用同一 checkpoint 检验候选覆盖瓶颈。只有 1000 pairs/2000 rollouts 完整落盘后，才回答“跨本体
是否提高成功率”。

## 8. 执行顺序与代码入口

```text
C 2000/8000 完整采集
  → C-only 五折 LOBO + N4 paired + ablation
  → B 100/400 e3/e4 source-only 补采与 immutable binding
  → C+B 五折 v9 LOBO
  → C+B 完整 N4 paired
  → 同 checkpoint 完整 N8 paired
```

- C collector：`scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py`
- B collector：`scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- B binding：`scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
- v9 trainer：`scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- 纯插件协议：`scripts/shared_event_critic_plugin_protocol_v1.py`
- N4 paired runner：`scripts/run_robotwin2_five_body_paired_success_v1.py`
- N8 runner：`scripts/run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- 完整增强 watcher：`scripts/watch_robotwin2_postformal_shared_head_upgrade_v1.py`

当前最重要的结果不是再加更多控制逻辑，而是先让 B 提供真实 e3/e4/恢复/多时域监督，再通过 N=8 扩大
可选动作覆盖，最后用 held-out 配对 ΔSR 判断共享头是否真正改善任务成功率。
