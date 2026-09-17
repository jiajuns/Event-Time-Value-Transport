# πRL Event-SMDP 实施边界

主线底座是官方 RLinf 的 π₀.₅ + πRL PPO/Flow-SDE；本仓库不重写 VLA、扩散/Flow 采样器或 PPO。官方代码在本机 `../RLinf-piRL`，并已同步到 HPC 的 `/data/user/leviccdong/EKSF/code/RLinf-piRL`。

## 当前 Role-Graph Event Observer

冻结 YOLOE 只输出 `object`、`target`、`gripper` 的候选框、置信度与 ROI。两层 GNN 仅对角色节点、轨迹、相对位置、尺寸比、IoU 与距离编码。它输出：

- `event_state`、七类 `event_posterior`（approach/grasp/lift/transport/align/place/unknown）；
- `event_progress`、`event_boundary_probability`、`event_uncertainty`；
- Event Value Critic 的 `event_value`；
- 原有关系、阶段、转移、目标和短期预测头作为辅助监督，绝不作为关系或事件真值。

已有 UMI 训练标签目前只覆盖旧的阶段/转移辅助头。因此新 Event-State/Event-Value 头已纳入模型 ABI 和预测文件，但在取得审核过的事件区间标签与在线 SMDP 回报之前，不得将其未训练输出用于 RL 或报告为价值结果。旧 CfC checkpoint 仅是历史对照，不能加载为本方案的 Event Observer。

## πRL 侧增量

RLinf 中的 `event_smdp_interventional` advantage 对每个连续事件计算
`R_j + gamma**D_j * V_E(e_{j+1}) - V_E(e_j)`。同一 simulator state 下的 Flow-SDE 多 action branch 用短 rollout 的 `R + gamma**H * V_E` 形成 influence；其 softmax 只在该 event 内分配 advantage。PPO actor loss 仍由 RLinf 原实现执行，新的 `event_actor` loss 刻意不把 Event Value target 误传给 π₀.₅ 原 VLA value head。

受控 branch 必须使用无损 `get_state` / `set_state`。先在 ManiSkill3 完成 oracle event 实验；当前 RobotWin 适配器没有该状态快照契约，不能以 reset 伪造干预。

## 实验顺序

1. 原样复现官方 π₀.₅ + πRL benchmark；Flow-Noise 与 Flow-SDE 使用独立配置，不混称。
2. ManiSkill3 oracle event：对比 GAE 与 Event-SMDP，报告 critic explained variance、success-vs-env-step AUC、N80/N90。
3. 加入四分支、10–20 step 的同态 intervention，对比普通 GAE、Event-SMDP、temporal event credit、interventional event-value credit。
4. 替换为 learned Event Encoder，报告 boundary F1、event F1、progress MAE 与性能差距。
5. 最后才扩展 CALVIN、LIBERO-Long/PRO 与 RobotWin；核心结论是等成功率下 online interaction 是否减少 30–50%，并报告完成链长度、credit ranking accuracy、多随机种子稳定性。
