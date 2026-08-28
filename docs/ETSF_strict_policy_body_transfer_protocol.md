# ETSF 第二策略 / 新本体严格迁移协议

## 结论

现有事件 critic 是可插拔的，但“能接入第二个 actor”不等于“已经跨策略迁移”，
“把 `body_id` 改成另一个整数”也不等于“已经跨本体迁移”。正式结论必须把两条轴拆开：

- **P-transfer**：同一任务、同一机械臂，只更换策略；
- **E-transfer**：同一任务、同一策略，只更换机械臂；
- 同时更换策略和机械臂只能作为最终 X-PE 扩展，不能替代上面两项可归因实验。

`scripts/verify_etsf_transfer_protocol.py` 将这组要求做成 fail-closed 协议。它只读取 JSON
逻辑身份和 CPU checkpoint，不导入 OpenVLA、SmolVLA 或 RoboTwin，不读取 HDF5 标签，也不
运行 GPU。现有 OpenVLA 密封确认 registry 被显式排除；迁移实验必须单独预注册目标域
adaptation / validation / confirmation seeds。

## 1. 当前哪些东西可以直接迁移

可以冻结复用的是共享事件半马尔可夫核心中的语义知识：事件词表和偏序、谓词语义、
next-event/reach/success/recovery/object heads、分布式持续时间输出形式、ensemble uncertainty
聚合和 guard 逻辑。前提是目标任务使用相同的 event-spec、对象坐标定义和标签口径。

以下内容不能因为维度“看起来兼容”就直接复用：

| 部分 | 当前 OpenVLA → SmolVLA | 需要的目标监督 |
|---|---|---|
| 状态 | OpenVLA hidden 4096-D 与 SmolVLA prefix 960-D 不同；旧 720-D expert hidden 还依赖 flow noise | 同状态配对蒸馏或目标 dense event 标签，训练小 `StateAdapter` |
| 动作 | chunk 为 `25×14` 与 `50×14`；归一化、执行 mask 和候选分布不同 | 目标候选 branch 的 action effect/success 排序，校准 `PolicyAdapter` |
| 策略 ID | 当前 policy embedding 是核心内参数，未见策略没有已训练行 | 源训练前预留一行，只允许该行用目标 adaptation split 更新 |
| 本体动作 | 原始关节量、控制器、动作尺度和可达集合不同 | `BodyActionAdapter` 的规范动作效果监督 |
| 本体时钟 | 同事件持续时间和控制频率不同 | 目标 adaptation rollout 的 duration/censoring，拟合 clock/`beta` |
| 谓词观测 | 名称相同不代表对象坐标系或视觉 detector 相同 | event-spec hash 相同才 identity；否则单独校准 `PredicateAdapter` |
| 当前事件 observer | simulator object pose 是 privileged 信息，不是可部署输入 | 主路径必须训练 RGB 或 actor-hidden observer；pose 只作 upper bound |
| guard | 源域 margin/uncertainty 阈值不能自动外推 | 只在目标 validation split 冻结，confirmation 只能打开一次 |

因此权重迁移不是“全部零样本”或“全部重训”二选一，而是：共享 event core 严格冻结，
目标端只训练小 adapter、clock 和预留的一个 policy/body embedding 行。目标数据不得对共享
核心做 TD 或长视界信用反传。

## 2. 当前数据为什么还不能证明迁移

现有 OpenVLA dense branch 是 `Piper + OpenVLA`，现有 SmolVLA actor/候选主要是
`Aloha + SmolVLA`。直接比较二者同时改变了 policy、body、state hook、action chunk 和 clock，
无法把变化归因于跨算法或跨本体。旧 SmolVLA schema-v2 的 720-D expert hidden 没有逐步
event/time/object 标签，也不能训练目标事件模型。

最短的第二策略路线是选择一个共同机械臂：

1. 在该机械臂上分别冻结可复现的 OpenVLA 与 SmolVLA actor；
2. SmolVLA 使用 schema-v5 collector 保存 denoise 前候选共享的 960-D prefix state；
3. 按完全独立的 target seed manifest 采 `50 adaptation + 20 validation + 50 confirmation`
   groups/任务；
4. 共享核心训练阶段必须完全排除目标策略，并预留目标 policy embedding 行；
5. 只用 adaptation 固定前缀 `N={0,5,10,20,50}` 训练 State/Policy adapters 和非特权
   event/predicate observer；主结果固定 `N=20`；
6. validation 选择 adapter checkpoint、temperature 和 guard，随后 confirmation 只评测一次。

最短的新本体路线同理，但必须保持 actor 算法不变，并增加 BodyActionAdapter 与 clock 适配。
Stage3 的四本体轨迹能支持事件/时钟机制诊断，但没有同一 policy 的候选动作分支，不能单独
证明动作条件 critic 在新本体上提高成功率。

## 3. 冻结 split 与数据门

协议的最小分组单位是：

```text
(task, policy, embodiment, requested_seed, resolved_seed)
```

source training、target adaptation、target validation、target confirmation 四者必须同时满足：

- requested seed 和 resolved seed 均不重叠；
- 每个文件的 SHA256 在查看标签前冻结；
- adaptation 按预注册顺序只取固定前缀，不能按成败或 uncertainty 选样；
- validation 每任务至少 20 groups；confirmation 每任务至少 50 groups；
- confirmation 不参与 adapter、scoring、temperature、distance 或 uncertainty guard 选择；
- P-transfer 的 body/task 完全相同；E-transfer 的 policy/task 完全相同。

## 4. 共享核心权重审计

当前模型把 `policy_embedding` 和 `body_embedding` 放在 action encoder 内。为了区分“小样本
adapter 适配”和不受审计地重训整个模型，正式源 checkpoint 必须在目标适配前拥有一个冻结
目标行：

```text
P-transfer: action_encoder.policy_embedding.weight[target_policy_id]
E-transfer: action_encoder.body_embedding.weight[target_body_id]
```

正式做法是在源训练前预留。`prepare_etsf_transfer_source_core.py` 只能把已有单 policy/body
checkpoint 转成一个**无目标数据、确定性词表初始化**：新行等于全部旧行的 float64 均值，旧
embedding 行和其他 tensor 保持 bit-exact，checkpoint 记录 parent SHA 与
`target_labels_read=false`。这个中间 checkpoint 明确输出
`vocabulary_preparation_requires_source_retraining / ready_for_protocol_freeze=false`，不能直接
进入 strict protocol。

随后必须在完全相同、冻结的 source manifest/split 上重训 expanded 模型，训练 batch 从不使用
reserved target id，也不读取任何 target 数据；独立 verifier 证明 source 参数确实变化、reserved
row 未变化、source manifest/split SHA 匹配，才输出
`source_core_ready_for_protocol_freeze`。strict protocol loader 会主动拒绝只有 expansion lineage
而没有 source-only retraining proof 的 checkpoint。

正式适配的协议审计仍要求 config、tensor key、shape、dtype 都不变；只允许冻结目标行变化，
其余行和全部 semantic/transition/event/time/object/uncertainty heads 必须 bit-exact。禁止在查看
目标数据后重新选择扩展初始化、容量或目标 ID。

```bash
python scripts/prepare_etsf_transfer_source_core.py expand \
  --source source_core.pt \
  --axis policy --target-name smolvla \
  --source-manifest source_manifest.json \
  --source-split source_split.json \
  --output source_core_reserved_smolvla.pt

# 使用与原 source core 相同的 manifest/split 重训后：
python scripts/prepare_etsf_transfer_source_core.py verify-source-retraining \
  --expanded source_core_reserved_smolvla.pt \
  --retrained source_core_reserved_smolvla_retrained.pt \
  --source-manifest source_manifest.json \
  --source-split source_split.json
```

典型命令：

```bash
python scripts/verify_etsf_transfer_protocol.py freeze \
  --draft transfer_protocol_draft.json \
  --checkpoint source_core_reserved_smolvla_retrained.pt \
  --output transfer_protocol.json

python scripts/verify_etsf_transfer_protocol.py audit-weights \
  --protocol transfer_protocol.json \
  --before source_core_with_reserved_target.pt \
  --after target_adapted_core.pt \
  --output transfer_weight_audit.json
```

外部 adapter 分别保存内容哈希与可训练参数量；结果中的 adapter 类型必须与 state/action/
predicate/clock 内容契约推导出的集合完全一致。

## 5. 必须报告的 baseline 与验收门

每个迁移轴都必须包含：原 actor、零样本 shared core、同 N/同容量 target-from-scratch、
full-finetune 上界、去因子化模型、actor-hidden observer，以及 simulator-pose privileged upper
bound。不能只和原 actor 比，也不能把 privileged pose 或 oracle candidate 当可部署模型结果。

主 `N=20` prediction gate 同时要求：

- 非特权 observer 的 current-event macro-F1 高于 frequency baseline，predicate macro-F1 高于
  constant baseline，有效覆盖率至少 90%；
- next-event macro-F1 高于 current-event 和 frequency baseline；
- success Brier 优于常数概率，ECE 不高于 0.10，并报告 PR-AUC；
- duration MAE 优于 event×body median；object-delta MAE 优于 zero-delta；
- within-group pair accuracy 大于 0.5；uncertainty AURC 优于随机覆盖顺序。

目标 confirmation 成功率门同时要求：

- 至少改变 10 个 episode，coverage 至少 10%；
- changed episodes 中 harmful rate 不超过 10%；
- paired success delta `>0` 且 episode bootstrap 95% 下界 `≥0`；
- 同 N 下成功率严格高于 target-from-scratch，并且可训练参数更少；
- 不劣于 no-factorization；confirmation 只访问一次。

confirmation 部署记录还必须包含 observer artifact SHA、`observer_mode=actor_hidden_observer`
或 `rgb_observer`、`privileged_inputs_used=false`。仅用 simulator pose 得到的分数只能进入
`privileged_pose_upper_bound` 行，协议会拒绝把它设置成 primary path。

配对区间不是由 child 随意填写：验证器用每个 episode 的 helpful/harmful/neutral 差值，以冻结
seed `20260827` 做 10,000 次 episode-paired percentile bootstrap，并从 helpful/harmful 重新
计算 exact McNemar/sign-test p。success counts、discordant counts、delta、CI 或 p 任一不一致都会
拒绝结果 artifact。

只有全部通过，结果 JSON 中 `action_ranking_authorized=true` 才会被验证器接受。指标不通过会
产出带原因的 monitor-only decision；手工把授权位改为 true 会直接报错。

```bash
python scripts/verify_etsf_transfer_protocol.py evaluate \
  --protocol transfer_protocol.json \
  --audit transfer_weight_audit.json \
  --results transfer_result_summary.json \
  --output transfer_acceptance_decision.json
```

## 6. 现阶段可准确表述的创新边界

代码已经支持一个安全的可插拔、跨策略/本体因子化接口，并新增了可证伪的迁移审计协议；
这使“共享事件核心 + 小 adapter”成为可执行实验，而不是概念描述。但在共同 body 的第二策略
以及同 policy 的新 body 上完成上述密封确认前，只能声称 **transfer-ready / monitor-safe**，
不能声称已经跨本体提高任务成功率。真正的创新证据来自三项联合结果：共享事件核心 bit-exact
冻结、目标样本/参数效率优于从头训练、密封闭环成功率置信下界非负。
