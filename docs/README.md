# docs 目录索引

## 当前先读这四份

1. `PROGRESS_20260831_1047_MODEL_HANDOFF.md`：最近一次完整的远程队列与接管快照；实时完成状态仍以服务器最终 receipt/report 为准。
2. `ETSF_ROBOTWIN2_SHARED_HEAD_V13_CAUSAL_BALANCE_ZH.md`：最新共享头结构与 v13 训练设计。
3. `ROBOTWIN2_CROSS_EMBODIMENT_EVALUATION_PROTOCOL_V3_ZH.md`：最终论文测评口径、指标公式和待补项。
4. `robotwin2_cross_embodiment_baseline_and_metrics_v1.md`：五折数据闭包、内部 baseline 和原始预注册设计；与 v3 冲突时以 v3 为准。

## A. 当前 RoboTwin2 五本体正式协议

### 数据、actor 与 branch

- `ETSF_RoboTwin2_EE16_actor_conversion_v1.md`
- `ETSF_RoboTwin2_move_can_pot_actor_episode_staging_v1.md`
- `ETSF_RoboTwin2_five_body_EE_candidate_branch_collection_v1.md`
- `ROBOTWIN2_SCRIPTED_EXPERT_ROOT_SUPPLEMENT_V1.md`
- `ETSF_RoboTwin2_move_can_pot_public_materialization_verifier_v1.md`

### 共享头与五折训练

- `ETSF_RoboTwin2_move_can_pot_five_body_LOBO_preregistration_v1.md`
- `ETSF_RoboTwin2_five_body_LOBO_shared_event_head_training_v1.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V13_CAUSAL_BALANCE_ZH.md`
- `ROBOTWIN2_WCM_STYLE_MATCHED_BASELINE_V1_ZH.md`

### 闭环与测评

- `ETSF_RoboTwin2_five_body_paired_success_execution_v1.md`
- `ETSF_RoboTwin2_five_body_LOBO_to_paired_success_watcher_v1.md`
- `ETSF_RoboTwin2_cross_embodiment_paired_success_evaluator_v1.md`
- `ROBOTWIN2_CROSS_EMBODIMENT_EVALUATION_PROTOCOL_V3_ZH.md`
- `robotwin2_cross_embodiment_baseline_and_metrics_v1.md`

## B. 共享头设计演化（历史，不代表当前入口）

以下文档保留用于说明设计为什么变化，当前实现以 v13 为准：

- `ETSF_RoboTwin2_shared_head_v6_terminal_consequence_design_zh.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V8_CROSS_EMBODIMENT_UPGRADE_ZH.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V9_CROSS_EMBODIMENT_DESIGN_ZH.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V10_CROSS_EMBODIMENT_DESIGN_ZH.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V12_COMPETING_RISKS_DURATION_ZH.md`
- `ETSF_ROBOTWIN2_SHARED_HEAD_V13_CAUSAL_BALANCE_ZH.md`
- `ETSF_shared_head_latest_design_zh.md`（总览；精确定义仍以版本化文档为准）

## C. OpenVLA / OpenVLA-OFT 线

入口文档：

- `ETSF_OpenVLA_OFT_full_technical_route.md`
- `ETSF_OpenVLA_progress_log.md`
- `ETSF_action_conditioned_event_world_model.md`
- `ETSF_event_critic_plugin.md`
- `ETSF_counterfactual_oof_protocol.md`
- `ETSF_counterfactual_sealed_evaluator.md`
- `ETSF_oof_prediction_diagnostics.md`
- `ETSF_policy_feature_action_bridge_contract.md`
- `ETSF_post_openvla_transfer_pipeline.md`

这部分记录 OpenVLA 训练、OOF 与插件化验证，不是当前 RoboTwin2 五本体 v13 的运行入口。

## D. SmolVLA / Piper 历史线

文件名包含 `SmolVLA`、`smolvla`、`Piper` 或 `schema6` 的文档属于：

- schema5/schema6 状态与对象姿态迁移；
- Piper 单本体 development/paired-success；
- source63 causal observer；
- evaluation400 与 autonomous watcher。

它们保留数据契约和运行事故的复现信息，但不能作为五本体 LOBO 结果引用。当前需要追溯时，先看：

- `ETSF_SmolVLA_Piper_schema6_adapter.md`
- `ETSF_SmolVLA_Piper_paired_task_success_protocol_v3.md`
- `ETSF_smolvla_shared_state_schema_v5.md`
- `ETSF_smolvla_source63_runtime_recovery.md`

## E. Stage0–Stage3 机制线

- `ETSF_agent_runbook.md`
- `ETSF_stage1_overnight.md`
- `ETSF_stage2_liquid_transport.md`
- `ETSF_stage3_factorized_transport.md`
- `ETSF_leave_one_body_out_transfer_protocol.md`
- `ETSF_multibody_canonical_event_world_model.md`
- `ETSF_novelty_and_baselines.md`

这些文档支撑事件结构、时间重参数化和早期 LOBO 机制结论，不表示当前远程实验状态。

## F. 进度快照与事故记录

- 最近一次完整交接：`PROGRESS_20260831_1047_MODEL_HANDOFF.md`；它不是实时监控结果。
- 其他 `PROGRESS_*`：只用于历史审计，不应用作当前 PID、路径、提交或结果来源。
- `ETSF_protocol_incident_20260827.md` 与 `ETSF_structured_event_audit_20260827.md`：协议事故/审计记录，不能删除后重新解释结果。

以后不为每次轮询新增进度文档。运行状态留在服务器 state/receipt；只有模型定义变化、远程接管或正式结果冻结时才更新交接文档。

## 保留与清理规则

| 类型 | 规则 |
| --- | --- |
| 当前协议、冻结哈希、数据/统计定义 | 保留并版本化 |
| 重大设计演化、事故审计 | 保留但标为历史 |
| 重复的即时状态输出、GPU 轮询、临时调试笔记 | 不写入 docs；放 `/tmp/etsf-*` |
| 训练日志、checkpoint、rollout、预测数组 | 不进入 Git；只在服务器输出根保存 |
| 已被新版本取代但仍被论文/哈希引用的文件 | 不物理移动；由本索引降级为历史 |

这样整理的目的，是让“当前入口”和“历史复现”明确分开，同时不因移动旧文件破坏已有链接、远端命令或协议哈希。
