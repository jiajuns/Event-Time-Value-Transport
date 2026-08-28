# OpenVLA 确认后的 ETSF 迁移流水线

> 2026-08-28 状态：本文件下方的 Aloha/schema-v5 方案保留为历史设计，不再是当前执行入口。当前严格主链是 r7h 原生五成员 source event world model → r8e Piper/UR5 LOBO → r9b Schema6 gate → target-development300 `80/30/190` → 五个 r7h-member-matched Piper adapters → formal190 evaluator/calibrator v2 → evaluation400 paired v2。当前远端证据和入口分别记录在 `ETSF_SmolVLA_Piper_schema6_autonomous_watcher.md`、`ETSF_SmolVLA_Piper_schema6_target_development300_preregistration.md` 与 `ETSF_SmolVLA_Piper_schema6_training_manifest_v3.md`；不得用本文件的旧 `50/20/50` 或 Aloha 双轴方案替代。

## 当前结论

服务器已具备 Aloha SmolVLA actor、LeRobot 0.4.4 环境和 RoboTwin 接口，但还不能立即做严格
迁移：现有 OpenVLA actor 是 Piper，SmolVLA actor 是 Aloha，策略和本体变化混杂；SmolVLA
现有数据全部为 schema-v1/v2，没有任何 960-D shared-prefix schema-v5 dense event/time/object
监督；审计版 schema-v5 collector 尚未部署到服务器代码目录。

只读盘点的内容寻址结果在
`artifacts/protocol/smolvla_aloha_transfer_asset_audit_20260827.json`。盘点没有读取 OpenVLA
fresh 标签，也没有启动或打断 GPU 作业。

## 已实现的自动控制链

`launch_etsf_post_openvla_transfer.py` 接收一个内容寻址的 argv-only plan，并等待现有
`launch_openvla_etsf_guarded_fresh_watcher.py` 的终态：

```text
OpenVLA watcher 未结束
        ↓ 只读 watcher_state.json，不打开 seed/HDF/result 标签
complete_fresh50_confirmed ?
        ├─ 否：迁移不启动，记录 transfer_not_started
        └─ 是
             ↓
       asset preflight
             ↓
       reserved vocabulary 确定性准备
             ↓
       相同 source manifest/split 重训
             ↓
       source-only provenance 验证
             ↓
       target shared-state smoke
             ↓
       adaptation / validation schema-v5 采集
             ↓
       冻结 strict transfer protocol
             ↓
       非特权 observer + privileged upper-bound
             ↓
       N={0,5,10,20,50} adapter curve
             ↓
       matched scratch / no-factorization / full-finetune
             ↓
       validation 冻结 observer、temperature、scoring、guard
        ├─ confirmation_forbidden：停止，confirmation 不可达
        └─ confirmation_authorized
             ↓
       primary-N 权重审计
             ↓
       独立 paired confirmation（≥50 groups/任务）
             ↓
       paired CI + strict transfer acceptance
```

每个 GPU 阶段都会重新检查设备名称包含 `RTX 4090` 且没有 compute PID，所有阶段严格串行。
当前 OpenVLA 采集/训练占卡期间，即使 watcher 已结束，迁移也不会抢占 GPU。

## Stage receipt 契约

plan 中不能放 shell 字符串，只能放 argv 数组；解释器和所有脚本均记录 SHA256。launcher 为
每个 child 注入：

```text
ETSF_TRANSFER_PLAN_SHA256
ETSF_TRANSFER_STUDY_ID
ETSF_TRANSFER_STAGE_ROLE
ETSF_TRANSFER_STAGE_RECEIPT
```

child 必须原子写入：

```json
{
  "format": "etsf_transfer_stage_receipt_v1",
  "study_id": "...",
  "plan_sha256": "...",
  "role": "train_transfer_n20",
  "status": "complete",
  "artifact_path": "/absolute/path/artifact",
  "artifact_sha256": "...",
  "labels_read": true
}
```

asset preflight、source expansion、interface smoke、protocol freeze、weight audit 必须报告
`labels_read=false`。OpenVLA watcher 只允许读 terminal state，自身始终记录
`openvla_confirmation_labels_read=false`。

launcher 不只信任 `asset_preflight` 的字符串状态：它会用
`verify_etsf_transfer_asset_preflight.py` 重新校验 actor/collector/observer 文件 SHA、schema-v5、
任务集合与迁移轴。policy transfer 必须“策略不同且本体相同”，embodiment transfer 必须“策略
相同且本体不同”；任一文件变更或 OpenVLA-Piper→SmolVLA-Aloha 双轴变化都会在任何 source
准备、采集或 GPU 阶段之前 fail closed。

## Source core 与目标 adapter

`prepare_etsf_transfer_source_core.py` 可对已有源 checkpoint 做一次 CPU-only、无目标数据的
确定性词表准备。旧 tensor 全部 bit-exact，新 policy/body 行用旧行 float64 均值初始化，输出
记录 parent file/state/config/contract SHA。但它明确不是可冻结 source core；流水线强制随后用
完全相同的 source manifest/split 重训，并运行 `verify-source-retraining`。只有 source 参数确实
变化、reserved row 未被 source batch 使用且保持不变、target 数据零访问时，才允许 protocol
freeze。manifest/split 的路径和 SHA 在 vocabulary preparation 当时就写入 lineage，事后换数据
会被拒绝。

`etsf_transfer_adapters.py` 实现：

- 低秩 `StateAdapter`：例如 SmolVLA 960-D → source-core 4096-D；
- 仿射 `Policy/Body ActionAdapter`：保留 native execution action，只转换 critic action-effect；
- embodiment-only `clock beta / step scale`；
- 仅目标 embedding 行的 gradient mask；每个 optimizer step 后恢复所有其他 core tensor；
- 部署 adapter 默认 monitor-only，只有独立确认通过后才可设置 calibrated/authorized。

这些接口解决“怎么适配”，但接口存在本身不是迁移证据。formal trainer 仍须用 schema-v5
固定前缀 N 数据监督 event/time/object/success/ranking，并输出 strict protocol 所需 artifact。

## Observer：privileged 与可部署路径必须分开

schema-v5 训练标签可以从 simulator object pose 推导，但在线 scorer 若继续直接读取 pose，只能
算 privileged upper-bound。迁移主路径必须使用：

- actor-hidden observer：从 StateAdapter 后的当前 hidden 预测 event/predicates；或
- RGB observer：从当前相机帧预测同一冻结谓词词表。

流水线强制 `train_actor_hidden_observer_n20` 与
`evaluate_privileged_pose_upper_bound_n20` 两个独立阶段。strict protocol 的 primary observer
只能是 `actor_hidden_observer` 或 `rgb_observer`；部署结果必须记录 observer artifact SHA 且
`privileged_inputs_used=false`。observer event/predicate F1 与 coverage 未过门时，后续闭环
confirmation 不授权。

## 当前仍缺的远端输入

要让流水线从“可验证编排”变成实际执行，必须先补齐：

1. 一个不混杂的 cell：OpenVLA-Aloha + SmolVLA-Aloha，或同一 VLA 的 source/target body；
2. 独立、reset-only 预注册的 target adaptation 50、validation 20、confirmation 50 scenes/任务；
3. 服务器部署并通过 960-D shared-prefix smoke；
4. schema-v5 target collector 输出；
5. 使用 `FrozenCoreTransferModel` 的 formal N-curve trainer、matched scratch trainer 和 receipt；
6. 非特权 observer 的 validation promotion artifact；
7. 只在 promotion 后运行的 paired online confirmation runner。

尤其是，第 5 项中的真实 `source_retrain_with_reserved_row` 训练器目前尚未实现；现有代码只有
vocabulary initializer、独立 proof verifier 和测试 fixture，fixture 不能作为实验 provenance。
因此 `launch_etsf_post_openvla_transfer.py` 当前是可执行、可 fail-closed 的编排骨架，不是一份
已经补齐所有 stage 命令、可直接产出迁移结论的服务器 plan。

因此目前应部署代码但保持迁移 watcher 未启动，或只让它等待 OpenVLA terminal state；资产
preflight 必须 fail closed，不能用现有 `OpenVLA-Piper → SmolVLA-Aloha` 数据绕过同轴门。
