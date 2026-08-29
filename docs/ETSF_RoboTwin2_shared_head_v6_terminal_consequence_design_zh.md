# ETSF RoboTwin2 跨本体共享头 v6：终局后果建模与完整效果协议

> 更新日期：2026-08-30。对应模型族
> `terminal_consequence_utility_shared_event_head_v6`。本文描述当前代码和即将执行的完整实验，
> 不把单元测试、单条轨迹或 critic 指标写成跨本体效果结论。

## 1. 当前结论

v6 已把共享头从“预测一步动作效果”改成“预测候选动作在冻结 actor 后续策略下的有限时域终局后果”。
代码已经具备五本体留一训练、best-of-4 重排和配对闭环执行接口，但目前还没有完成全部
`5 body × 2 condition × 40 query × 5 seed × 4 candidate = 8000 branch`，因此现在不能声称
共享头已经提高跨本体成功率。

最终只用以下主指标判断模型是否有效：

\[
\Delta SR=SR(\text{frozen actor + shared head rerank})-SR(\text{frozen actor})
\]

并同时报告配对阶段进度增益、McNemar 检验和按 seed 聚类的 95% 置信区间。AUC、Brier、NLL、
MAE 和覆盖率只用于解释共享头为什么有效或为什么失败，不能替代任务成功率。

## 2. 为什么必须从一步效果升级到终局后果

现有真实分支已经出现过这样的候选对：二者的 `post_event`、`next_event`、duration、即时对象变化和
success 标签都相同，但冻结 actor 继续执行后，最终目标进度相差约 25 cm。只看候选后的前五个动作，
共享头无法区分这两个候选；这不是扩大 MLP 或增加接受门能够解决的问题，而是监督时间域缺失。

v6 因此显式增加：

\[
p(E_H\mid s_t,u_t,e_t,A_t),\qquad
p(G_H\mid s_t,u_t,e_t,A_t),
\]

其中：

- \(E_H\) 是从当前根状态到正式 episode 预算结束期间达到过的最大规范事件；
- \(G_H\) 是根状态到分支终止时的目标距离进展，允许为负；
- \(A_t\) 是打分时已知的剩余 action budget；
- 后续动作由同一个冻结 actor 产生，所以模型学习的是“候选动作 + 固定 continuation policy”的后果，
  不是脱离策略的无限时域动力学。

## 3. 跨本体输入合同

五个本体 `aloha-agilex / arx-x5 / franka / piper / ur5` 先被解析为相同的任务空间表示：

- canonical state：27 维，包括 object→goal、左右 EE→object、对象位移、夹爪、对象姿态、当前事件
  和四个事件谓词；
- canonical action：14 维，左右 EE 各自的 `Δxyz + Δaxis-angle + Δgripper`；
- 事件词表：`e0 / e12 / e3 / e4 / eK`；
- `event_age_seconds`：当前事件已经持续的真实物理秒；
- `remaining_action_budget`：正式 200-action episode 中尚可执行的动作数。

共享头只有一个 state/action stem 和一套权重，不为 held-out body 建 embedding、随机 adapter 或独立
输出头。不同机器人只通过无标签解析式运动学适配到上述 canonical space。因此 LOBO 中目标本体没有
任何可训练专属参数，这才构成严格的跨本体迁移。该设计不保证任意新机器人自动零样本成功；新本体
仍必须能可靠提供相同语义的 state/action/event 接口。

## 4. v6 网络结构

基础事件世界模型得到动作条件表示 \(\tau_t\)，并预测即时后果：

- `post_event_logits`；
- `next_event_logits`；
- `success_logit`；
- `recovery_logit`；
- `duration_selected_log_mean/log_scale`；
- `object_delta_mean/log_scale`。

终局上下文只使用打分时可得变量：

\[
c_t=\operatorname{MLP}(\log(1+\text{event age}),
                        \log(1+\text{remaining budget})),
\]

\[
h_H=\tau_t+c_t.
\]

随后输出五类 terminal-event logits，以及 Student-t(3) terminal-goal-progress 的 mean/scale。由于
`terminal_max_event` 定义为包含当前根状态的最大事件，低于当前事件的类别在结构上不可能，代码只
对这些类别施加固定的物理支持约束；这不是根据置信度回退动作的部署门控。

Recovery 也改为可识别的联合量：

\[
p_{regress}=\sum_{k<e_t}p(e_{post}=k),
\]

\[
p_{joint\ recovery}=p_{regress}\,
\sigma(\ell_{recovery}).
\]

所以在 `e0` 上二者严格为零，不再把无适用性的裸 recovery 概率输入排序头。

## 5. 排序头只组合显式预测后果

候选 utility 固定读取 36 维特征：

| 特征 | 维数 |
|---|---:|
| post-event probability | 5 |
| next-event probability | 5 |
| success probability | 1 |
| regression probability | 1 |
| joint recovery probability | 1 |
| duration `log1p` mean / scale | 2 |
| object SE(3) mean / scale | 12 |
| 即时目标进度 / 径向不确定性 | 2 |
| terminal-event probability | 5 |
| terminal-goal-progress mean / scale | 2 |
| 合计 | 36 |

整块 36 维特征在进入 utility MLP 前 `detach`。因此 listwise ranking 只能学习“如何组合已经受 proper
loss 约束的预测后果”，不能把 transition hidden 偷改成一个不可解释的自由 critic。部署时始终在
冻结 actor 给出的四个候选中直接选择最高分，不设置效果阈值、不根据 critic 指标授权候选，也不修改
actor 权重。

## 6. 监督数据与损失

训练监督不是官方人工标签文件。actor 先用公开 RoboTwin2 `move_can_pot` 演示训练，然后在公开
RoboTwin2 模拟器中，从同一个根状态执行四个 actor 候选并继续运行到成功或正式 action limit，
自动得到完整 consequence labels。

每个候选保存：即时/下一事件、duration 及删失状态、success、recovery、对象 SE(3) 变化、
`terminal_max_event_id`（从候选根状态起，不混入根状态之前的历史最大事件）、terminal goal
distance/progress、终止原因、事件年龄和剩余预算。正常成功、
正式 action limit 是正式有限时域结果。RoboTwin 返回 CuRobo `Fail` 而不抛异常时也属于真实策略
后果；actor generation、动作执行 Python 异常、观测、渲染、CUDA 或根状态恢复异常必须使整个
四候选 decision 作废，不能被写成失败样本。

训练有两个隔离的数据流：

1. uniform complete-decision stream：训练未重权的 proper success/event likelihood、稳定右删失
   lognormal duration、Student-t(3) object effect、terminal-event CE 和 terminal-progress
   Student-t(3) NLL；
2. balanced rank-only stream：训练四候选 listwise utility。它可过采样稀有 mixed-success decision，
   但不会反向更新 world/terminal heads，也不会改变概率先验。

terminal-event 和 terminal-progress 的初始 loss 权重均为 0.5。全失败 decision 的 dense rank 只占
0.1 权重：先比较达到的最大事件，再在相同最大事件内比较终局目标进度。mixed-success decision
直接训练把概率质量分给成功候选。

## 7. 保证四候选确实来自同一个根状态

旧实现通过 reset 后重放动作前缀到 query10/20/30，但 CuRobo 重规划的物理步数可能变化，导致四个
候选并非同根干预。v3 collector 改为：

1. 前缀只执行一次并显式保存 keyed SAPIEN 状态、任务计数、机器人缓存、渲染状态和四类 RNG；
2. 生成候选和执行每个候选都创建独立 fresh scene；
3. 恢复 qpos/qvel/qf、根位姿/速度、drive targets、动态刚体和 task/RNG；
4. 每个 actor query 前统一执行一个计入物理时钟的 raw scene step；fresh-scene 根恢复后的这一步
   同时以相同方式重建 contact cache；
5. 对规范化后的完整状态做严格 SHA，并校验对象位姿和 EE 状态误差不超过 `2e-5`。

正式 paired runner 在每次候选生成前也执行相同的一个 raw scene step，并对 baseline/ETSF 两种方法
完全一致地计入物理时间而不增加 action count，保证训练和部署的 critic 评分时刻同义。

PhysX 的 `qacc` 是上一 solver step 的派生缓存，SAPIEN 3 的 fresh scene 无法原样写回。它仍保存在
原始 provenance snapshot 中，但不参加 canonicalization 前的“可恢复状态”哈希；统一物理步重新生成
后，它重新进入完整严格哈希。其他独立物理状态不放宽。

## 8. 完整 LOBO 与闭环评估

完整开发数据固定为 2000 个 decision、8000 个 branch。每个 body/condition 的 200 个 decision
均匀覆盖 query 0--39，每个 query 使用五个 development seed，因此 remaining budget 的
200、195、……、5 全部有训练支持。每折只用四个 source bodies 训练五成员
ensemble，held-out body 不参与 normalization、训练、选步或消融选择。source validation 的同一
requested seed 下全部 query 保持在同一 split，避免相邻状态泄漏。

离线每折报告：

- terminal event：NLL、Brier、accuracy、macro-F1、ordinal error；
- terminal progress：MAE、RMSE、Student-t mixture NLL、90% coverage；
- success/recovery/regression：Brier、NLL、AUROC/AP（有支持时）；
- one-deviation branch：selected/oracle success、相对 candidate 0 的分支成功增益、pairwise
  accuracy、阶段/目标进度 regret；
- uncertainty：五成员分歧对应的 risk-coverage/AURC。

离线分支估计的是“当前只偏离 actor 一次，之后回到冻结 actor”的 \(Q^{actor}\) 改善，不是递归
重排整条 episode 的闭环 `ΔSR`，代码和报告不再把二者同名。按 \(Q^{actor}\) 贪心是策略改进的
训练依据，但近似误差与状态分布漂移仍必须由闭环实验验证。

最终确认实验使用未参与训练的正式 seed。相同初始 seed 分别执行 candidate 0 baseline 和 v6
best-of-4，报告二值成功与阶段进度的配对差值。只有五个 held-out body 的等权宏平均、各本体结果和
95% CI 都完整落盘后，才回答“能否跨本体提高任务成功率”。

## 9. 消融和可插拔边界

完整消融固定四种：

- `success_only`：只按 proper success logit；
- `no_time_duration`：清零 duration 排序特征，并禁止 terminal heads 读取 event age/remaining budget；
- `no_object_effect`：去掉 object、即时 goal 和 terminal goal，但保留 terminal event；
- `full`：36 维完整 v6。

模块接口是 `state + event context + N canonical action candidates -> N scores`，所以可接 OpenVLA、
SmolVLA 或其他能够给出多个任务空间动作候选的策略。迁移时不直接复用某个策略的 raw joint action：
必须先实现同语义 canonical action adapter，并用该冻结策略产生后果监督。可插拔描述的是推理接口和
actor 权重隔离，不代表一个 checkpoint 对所有策略、任务和本体无需验证即可通用。

## 10. 代码入口

- 分支采集：`scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py`
- LOBO 训练：`scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- 完整离线消融：`scripts/run_robotwin2_five_body_lobo_offline_ablation_v1.py`
- 配对闭环执行：`scripts/run_robotwin2_five_body_paired_success_v1.py`
- actor→采集 watcher：`scripts/watch_robotwin2_ee16_actor_to_five_body_branches_v1.py`
- 采集→LOBO watcher：`scripts/watch_robotwin2_five_body_branches_to_lobo_training_v1.py`

当前优先级不再是增加网络分支或部署门控，而是完成同根采集、五折 LOBO 和配对闭环，用真实
`ΔSR` 决定 v6 的终局后果建模是否值得保留。
