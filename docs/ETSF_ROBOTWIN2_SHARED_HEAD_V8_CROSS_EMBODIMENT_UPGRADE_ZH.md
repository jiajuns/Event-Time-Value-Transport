# ETSF RoboTwin2 跨本体共享头 v8：事件后果预测、临界成功补充与闭环提升协议

> 更新日期：2026-08-31。对应模型族
> `terminal_consequence_utility_shared_event_head_v8`。本文只描述当前实现和已经冻结的完整实验；在
> held-out 配对闭环完成前，不把单元测试、AUC、MAE、Brier 或候选 oracle 写成跨本体成功率结论。

## 1. 要解决的问题

共享头不是另一个动作策略，而是冻结 actor 外部的候选动作价值模块：

```text
当前 canonical 状态 s_t、事件 e_t、事件年龄 a_t、剩余预算 H_t
                         +
              actor 给出的 N 个动作候选 u_t^1...u_t^N
                         ↓
              共享事件/后果模型逐候选预测
                         ↓
                选择风险调整 utility 最高者
```

它必须同时满足四个目标：

1. 对事件转移、持续时间、成功/失败/恢复、对象状态变化和不确定性作可检验预测；
2. 不为目标机器人增加专属可训练参数，能够用严格 leave-one-body-out 证明跨本体；
3. 保持 actor 冻结，可接在能输出多个任务空间动作候选的 OpenVLA、SmolVLA 或其他策略外部；
4. 最终确实提高 held-out 本体的任务成功率，而不只是在离线 critic 指标上更好。

## 2. 当前模型接口与跨本体边界

五个本体 `aloha-agilex / arx-x5 / franka / piper / ur5` 被解析到相同任务空间：

- 27-D canonical state：object→goal、左右 EE→object、对象位移、夹爪、对象四元数、当前事件 one-hot
  与四个解析式谓词；
- 14-D canonical action effect：左右 EE 各自的 `Δxyz + Δaxis-angle + Δgripper`；
- 事件词表：`e0 / e12 / e3 / e4 / eK`；
- `event_age_seconds`：当前事件已经持续的模拟器物理秒；
- `remaining_action_budget`：200-action episode 中打分前剩余动作数。

模型只有一套 state/action stem、一个共享 body row 和一套输出头。held-out body 不参与归一化、训练、
checkpoint 选择或 calibration，也没有 embedding、adapter MLP 或独立 head。这证明的是“同任务、同
canonical 语义下的跨本体迁移”。如果换成新任务、不同事件本体或 raw joint action，必须先实现对应
的无标签 canonical adapter，不能直接把当前 checkpoint 宣称为无条件通用。

推理接口为：

```text
(state27, event_age, remaining_budget, N × canonical_action14[H]) → N 个分数
```

因此模块在工程上可插拔，且 actor 权重不被修改；但可插拔不等于无需在新 actor 分布上验证。

## 3. v8 预测的世界后果

对每个候选，基础世界模型预测：

- `post_event_logits`：执行候选前五个动作后的事件；
- `next_event_logits`：下一事件；
- `duration_selected_log_mean/log_scale`：到事件边界的物理持续时间；
- `object_delta_mean/log_scale`：对象 SE(3) 变化；
- `regression_probability` 与 `joint_recovery_probability`；
- `terminal_event_logits`：候选执行后由同一个 frozen actor continuation，在剩余预算内达到的最大事件；
- `terminal_goal_progress_mean/log_scale`：根到有限时域终止的目标距离进展；
- `success_probability`：严格定义为 terminal event 分布中的 `p(eK)`。

终局上下文为：

\[
c_t=\operatorname{MLP}(\log(1+a_t),\log(1+H_t)),\qquad
h_H=h_{transition}+c_t.
\]

这使同一状态/动作在 H=10 和 H=200 时可以得到不同成功概率。`success` 与 terminal `eK` 使用同一
概率分布，不再允许“高成功率但低 eK 概率”的互相矛盾输出。低于当前事件的 terminal 类别被固定
物理支持屏蔽；这属于事件定义，不是动作回退 gate。

Recovery 是可识别的联合量：

\[
p(regress)=\sum_{k<e_t}p(e_{post}=k),
\]

\[
p(joint\ recovery)=p(regress)\,\sigma(\ell_{recovery}).
\]

在 e0 上没有更低事件，因此回退和联合恢复概率严格为零。

## 4. 排序 utility 不再是自由黑盒 critic

模型把 proper heads 的输出压缩为九个 `[0,1]` consequence features：

| 类型 | 特征 |
|---|---|
| 收益 | post expected stage、next-event advance rate、success probability |
| 稳定性 | no-unrecovered-regression probability |
| 短期对象效果 | short goal-progress benefit、short uncertainty risk |
| 终局后果 | terminal expected stage、terminal goal benefit、terminal uncertainty risk |

九维特征在进入排序头前整体 `detach`。收益特征只允许非负 softmax 权重，不确定性只允许非正有界
权重：

\[
U=\sum_i softmax(\alpha)_i b_i-\sum_j sigmoid(\beta)_j r_j.
\]

因此 listwise ranking 学的是“如何组合已经由 proper loss 约束的后果”，不能利用排序标签把事件、
时长或对象预测偷偷改写成不可解释 latent score。world 参数和 utility 参数分别裁剪梯度。

为了让终局语义在真实候选比较上更敏感，v8 还有一个受限 semantic comparative loss；它可更新 terminal
predictors，但每个 head 以及整个 shared world 的相对梯度预算均固定不超过 proper world gradient 的
0.1。它不是额外候选 gate。

五成员 ensemble 的部署分数固定为：

\[
score_i=\operatorname{mean}_m U_{m,i}-0.5\operatorname{std}_{m,pop}U_{m,i}.
\]

直接执行最高分候选，平分时取最低 index；没有置信度阈值、授权开关或回退逻辑。

## 5. 两条监督数据流

### 5.1 C：正式 actor-prefix 分支数据

正式数据为五本体、两条件、40 个 query、每 query 五个 seed、每根四候选：

\[
5\times2\times40\times5=2000\ decisions,
\]

\[
2000\times4=8000\ branches.
\]

每个候选从完全相同的可恢复 SAPIEN 根状态执行前五个 actor 动作，然后继续由同一个 frozen actor 到
成功或 action limit。它提供 on-policy 状态分布、概率校准、source validation、checkpoint 选择和
排序监督，是主数据流。

截至 2026-08-30 的 553-decision 审计：

- 242/553（43.8%）根具有真实阶段/目标进度差异；
- 平均 candidate oracle 阶段增益 +0.00904，目标距离进展 +0.04165 m；
- 只有一个根的四个 branch 成功，mixed-success 根为 0；
- candidate-0 和 N=4 success oracle 都为 1/553。

这说明 C 已有事件进度排序信息，但尚未观测到 N=4 二值成功的可选择空间。该 1/553 是 one-deviation
branch 比例，不是整条策略 episode 成功率。

### 5.2 B：scripted e3/e4 root + frozen actor 后果补充

直接把官方 expert 成功轨迹当作 actor 的成功标签会改变目标分布，因此没有这样做。B 的协议是：

1. public `move_can_pot.play_once` 只推进到第一次非终止 e3/e4 物理样本；
2. 保存根后立即结束 expert 语义；
3. 从该根启动新的 frozen actor 分支，四候选和 continuation 都与 C 使用同一 actor；
4. 成功/失败、事件、时长和对象变化全部来自这个 actor 分支，不读取 expert 最终结果；
5. horizon 在任何 rollout 前按五个固定 seed 标签盲绑定为 `10/25/50/100/200`。

完整规模为五本体 × 两条件 × e3/e4 × 五 seed = 100 decisions / 400 branches。B 只补足临界成功边界，
不追加到 C root，也不声称是正式 actor-prefix 状态分布。

每个 outer fold 只打开四个 source-body B manifests 和 payload。held-out B manifest 与 NPZ 都 zero-open。
B 以固定 `lambda=0.25` 只训练 proper multitask、robust object、terminal event/progress losses：

- 不参与 state/action normalization 或 baseline fit；
- 不进入 rank/utility loss；
- 不进入 source validation、checkpoint selection 或 calibration；
- 不做 class balance、不合成标签；
- 五成员只做普通 logical-group Poisson bootstrap。

这样 B 改善事件后果表示和正负成功边界，但所有模型选择仍由 C source validation 决定，降低 targeted
near-success 数据造成校准偏移的风险。

## 6. N=4 与 N=8 为什么分开

当前正式协议保留原 N=4，不改历史基线。另建 postformal N=8 闭环，因为共享头只有在 actor 候选池中
真实存在更好动作时才能提高成功率。

- N=4：正式 baseline 和增强 shared head 的主可比结果；
- N=8：同一个增强 checkpoint，只沿候选轴扩展，检验候选覆盖是不是瓶颈；
- 可选 N=16：后续预算允许时使用；
- baseline 永远执行 raw flow proposal 0，不调用 critic；
- raw proposal 0 的噪声在 N=4/8/16 中不变；
- 默认 raw proposal 数等于 N，保持 actor 原顺序；
- 显式 oversampling 最多 2N 时，才使用 outcome/event/critic-blind 的 canonical-effect farthest-point
  sampling，并固定保留 raw 0。

两种配对方法要求相同 requested seed、完整 observable reset snapshot、query-0 canonical snapshot、raw
proposals 与 retained pool commitment。执行第一个不同动作后状态自然分叉，因此后续候选池不要求相同；
要求后续相同反而不符合闭环策略比较。

## 7. 严格 LOBO 和最终成功标准

每个 fold：四个 source bodies 训练，另一个 body held out；五个 ensemble member 使用独立完整 decision
Poisson bootstrap。source train/validation 按 `(body, condition, requested_seed)` 分组，所有 query 保持同
一侧，并标签盲覆盖完整 H=200...5 网格。

离线诊断包括 event F1/NLL、duration MAE/NLL、object Student-t NLL、success Brier/NLL、terminal event/
progress、pairwise ranking 和 uncertainty。这些用于定位机制，不构成迁移成功证明。

最终完整评估对每个 held-out body、clean/randomized 和 100 个预注册 seed 执行两次：

```text
actor baseline                     1000 rollouts
actor + 对应 held-out LOBO head     1000 rollouts
```

主指标：

\[
\Delta SR=SR(actor+head)-SR(actor).
\]

同时报告：

- 二值成功率和配对 bootstrap 95% CI；
- McNemar exact two-sided p；
- 阶段进度 `0 / 0.25 / 0.5 / 0.75 / 1.0` 的配对差和 95% CI；
- 每个 held-out body、body×condition 和等权总体结果。

增强 N=4/N=8 runner 必须读取同一个 supplement binding SHA；五个 fold 若混入 C-only、不同 supplement
或缺失 B，执行前直接失败。只有完整 1000 pairs / 2000 rollouts 落盘后才回答“跨本体是否提高成功率”。

## 8. 完整执行顺序

现有 C-only 正式流水线保持不变：

```text
2000/8000 C collection
  → C-only 五折 LOBO
  → C-only N=4 paired success
  → 完整四变体 ablation
```

新的增强 watcher 等上述任务结束并确认 4090 空闲后运行：

```text
100/400 B collection
  → immutable supplement binding
  → C+B 五折 LOBO
  → C+B 完整 N=4 paired success
  → C+B 完整 N=8 paired success
```

两条流水线使用独立 root、checkpoint、报告和只读代码目录；不覆盖 C-only 结果，也不在当前正式采集
过程中抢占 GPU。

## 9. 代码入口

- 正式 C collector：`scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py`
- B expert-root collector：`scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- B binding materializer：`scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
- v8 LOBO trainer：`scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- C/C+B LOBO watcher：`scripts/watch_robotwin2_five_body_branches_to_lobo_training_v1.py`
- N=4 paired runner：`scripts/run_robotwin2_five_body_paired_success_v1.py`
- N=8/16 postformal runner：`scripts/run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- 完整增强流水线 watcher：`scripts/watch_robotwin2_postformal_shared_head_upgrade_v1.py`

当前改进方向已经固定为“增加真实临界后果监督 + 增加 actor 候选覆盖 + 完整 held-out 闭环验证”，而不是
继续增加不改变可选动作集合的 gate 或用小规模测试代替成功率。
