# ETSF 结构化事件审计与 v2 监督契约（2026-08-27）

## 1. 审计对象和边界

本报告只读审计远端既有三种子输出：

```text
/home/user/etsf_openvla_event_world_model_move_can_pot_balanced_20260827
```

审计读取了 `query_transitions.pt`、三个 `event_world_model_best.pt`、训练日志和
训练/验证 split；没有读取 sealed test 的样本或指标，没有启动远端训练，也没有覆盖
既有 checkpoint。下述改造只改变后续新输出目录的训练契约。

## 2. v1 checkpoint 的真实事件混淆

验证集 186 条转移的旧 `next_event_id` 分布为：

| event | e0 | e12 | e3 | e4 | eK |
|---|---:|---:|---:|---:|---:|
| 标签数 | 8 | 174 | 2 | 0 | 2 |

三模型的绝对事件 macro-F1 分别为 `0.235795 / 0.236544 / 0.235795`。概率平均
ensemble 的混淆矩阵（行是真值，列是预测）为：

```text
       pred e0 e12 e3 e4 eK
true e0      0   8  0  0  0
true e12     7 167  0  0  0
true e3      0   2  0  0  0
true e4      0   0  0  0  0
true eK      0   2  0  0  0
```

即 ensemble 只输出 7 次 `e0` 和 179 次 `e12`，从未预测 `e3/e4/eK`；约 0.89 的
accuracy 是多数类结果，不能作为准确事件建模证据。

更关键的是：验证集中 134 条 `next_event_id == current_event_id` 的样本全部是
`duration_observed == 0` 的右删失样本。它们只表示“episode 截止时尚未观察到下一
事件”，不是真实 self-loop。v1 用这些样本做绝对事件交叉熵，混淆了“是否可达”和
“到达哪个事件”。

三个 best checkpoint（而非 early-stop 时的 latest 状态）均值为：

| 指标 | 三种子均值 |
|---|---:|
| reach AUC | 0.971824 |
| success AUC | 0.649811 |
| observed duration MAE | 15.8507 steps |
| future semantic cosine | 0.931359 |
| object delta MAE | 0.058645 m |
| absolute event macro-F1 | 0.236045 |

旧 `training_summary.json` 的 `last_validation` 是 early-stop 时的 latest 指标，不是
best checkpoint 指标，因而出现 96--109 steps 的 duration MAE。新训练脚本同时写
`best_validation` 和 `last_validation`，避免再次混报。

## 3. v2 标签分解

新模型将三个不同问题分开监督：

1. **动作块局部事件效果**：从 query step 到 `min(query+H, terminal)` 的动态
   post-event，所有 factual transition 都有完整监督。
2. **后续事件目的地**：`p(next reached event | reach)`，只对
   `duration_observed==1` 的样本计算 CE；右删失样本不再伪造成 stay。
3. **事件持续时间**：保留 log-normal observed/right-censored likelihood。

动态事件不再只由 first-hit 历史决定，而由每一步原子谓词决定：

```text
moved, lifted, near_goal, stationary, success
```

phase 优先级为：

```text
success > stationary > near_goal > (moved | lifted) > e0
```

其中 `lifted/near_goal/stationary` 可以由真变假，但单个谓词 down-flip 不再直接定义
regress。operational regress 必须是动态 phase 低于 drop 前历史 peak，并连续保持至少 3 个
simulator state；这会排除放下物体但仍停留在 e12，以及 1--2 步阈值抖动。success 仅在
成功 terminal step 为真，不能把 episode 最终成功泄漏到先前 query。

在现有 factual train/validation 上重新派生后的分布为：

| split | dynamic post `[e0,e12,e3,e4,eK]` | relative `[stay,advance,skip,regress]` |
|---|---|---|
| train | `[94,641,2,1,12]` | `[636,100,13,1]` |
| validation | `[25,158,0,0,3]` | `[159,24,3,0]` |

这修正了标签语义，但没有凭空增加晚期数据：train 只有 1 条 regress，validation 没有
regress/e3/e4。因此 factual v2 仍是预训练，必须依靠 schema-v5 完整分支轨迹增加
near-goal、stationary、regress 和 recovery 覆盖。

上述稀缺性仅指 150 条旧 factual rollout 的固定 train/validation split，不代表新采集的
schema-v4/v5 没有正例。截至同日对**实采前 88 个 schema-v4/v5 候选分支**重新审计：

```text
旧 predicate-down/up 宽松定义: regress 47 / recovery 42（废弃）
dynamic phase 真正下降:       21
phase 下降持续 >= 3 states:   20
恢复旧 peak 持续 >= 3 states，
或之后 terminal success/eK:  12
```

当前标签契约为：regress 只由持续 phase drop 派生；recovery 要求其后回到 drop 前 peak
并持续至少 3 states，或其后明确到达 terminal success/eK。terminal success 是恢复的
允许终点，不是从 episode success 反推之前发生过 regress；没有先满足持续 phase drop
就不能标 recovery。阈值由训练 CLI `--regression-persistence-steps` 固化进 checkpoint
contract 和 `data_audit.json`。正式结论仍须在按 logical group 切分后的 validation 中
分别报告正例数和指标，不能把全体前 88 条统计当成验证结果。

## 4. 模型输出与约束

`structured_events=True` 时新增：

```text
relative_transition_logits: [B,4]
post_predicate_logits:       [B,5]
post_predicate_probability:  [B,5]
predicate_delta:             [B,5]
next_reached_event_logits:   [B,E]
```

旧 key `next_event_logits [B,E]` 保留，但表示由相对转移和条件目的地重构的 post-event
分布。每个绝对目的地唯一映射到 stay/advance/skip/regress；没有对应目的地的类别概率
被硬置零。`allow_event_regress=False` 可用于事件必须单调的任务，确保低于当前 phase 的
概率严格为零；动态任务保持为 True。

新监督损失为：

```text
CE(structured post event)
+ CE(relative transition; train support < 5 的类别不优化)
+ BCE(post predicates)
+ CE(next reached event; observed-only)
+ censored duration NLL
+ success/outcome/object/future-latent losses
```

事实数据没有 operational recovery 标签，因此结构化 factual 训练仅在 outcome 的
failure/success 子空间优化；不会把所有样本硬标成 recovery 负例。只有 schema-v4/v5 从
完整轨迹明确派生“phase drop 持续至少 3 states，随后恢复到旧 peak 持续至少 3 states
或到达 terminal success/eK”之后，才能启用 `recovery_supervised=True`。

恢复使用既有 `outcome_logits[:, 2]` 表达，不另建重复 head。反事实微调的监督契约是：

- schema-v4/v5 且 recovery 支持数达到训练阈值：用三分类
  `failure/success/recovery`；
- v2/v3 或无完整轨迹的样本：只在 `outcome_logits[:, :2]` 上训练
  failure/success，不能充当 recovery 负例；
- recovery 支持不足：保持 `recovery_supervised=False`，checkpoint 与审计明确写
  `unsupported`，部署不读取随机 recovery 概率。

## 5. 兼容性与验证

- `structured_events=False` 不注册新参数，v1 checkpoint 可 `strict=True` 加载。
- `next_event_logits`、clock、success、object、future latent 等既有 key 保留。
- 新 factual cache schema 为 3；旧 cache 必须显式重建，避免静默混用错误标签或 sealed
  test transitions。
- 新 ensemble launcher 默认写入新目录
  `/home/user/etsf_openvla_structured_event_world_model_move_can_pot_sealed_schema3_20260827`，
  不会覆盖 v1 或严格契约升级前的部分训练。
- 本机仅运行 CPU 模型/标签测试，没有进行真实数据训练：当前仓库全套测试共
  `63 passed`。

## 6. 后续验收门

在 schema-v5 分支数据训练前，不能声称事件预测已改善。正式训练必须报告：

1. dynamic post-event 和 relative transition 的 per-class confusion/F1；
2. 每个 predicate 的 positive count、AUC、F1、Brier；
3. observed-only next-reached-event macro-F1；
4. relative/absolute consistency（结构约束应为 1）；
5. regress/recovery 的独立支持数，验证集无正例时写 `unsupported`；
6. 最终同状态候选 pair accuracy、guard 覆盖率和真实 success 增量。

只有 sealed policy evaluation 显示成功率提高，才能从“更合理的事件监督”升级为
“改善任务成功率”的结论。

## 7. 正式事实预训练队列严格性复审

`run_openvla_etsf_event_world_model_ensemble.sh` 与 factual trainer 的正式契约如下：

- 固定使用 `--event-mode structured`，并显式传入
  `/home/user/etsf_stage2_run_20260825/event_spec.json`；每次 cache build/reuse 都重新计算
  SHA-256，并与 rollout manifest、cache 和 checkpoint contract 同时核对。
- 使用专用 `query_transitions_schema3_train_validation_only.pt`，不读取或覆盖旧通用
  `query_transitions.pt`。cache 只含 train+validation seed；sealed test 只保留
  path/seed/metadata/raw SHA-256。
- checkpoint contract 绑定训练 seed、split seed 列表、rollout manifest SHA、event-spec
  SHA 和 predicate derivation/calibration；因此 seed 目录错配不能静默 resume。
- resume 要求 latest/best 的 contract、best step 和 best score 完全一致；若旧目录不兼容，
  launcher fail-closed 并要求新 `OUTPUT_ROOT`，不会覆盖旧文件。
- “三 seed 完成”不再由 grep JSON status 决定。只在恰好三个不同 seed 的 summary、best、
  latest、schema-3 cache 全部通过 `verify_openvla_etsf_factual_run.py` 后打印
  `ENSEMBLE_TRAINING_COMPLETE`。
- best 只由 validation selection score 更新；validation 没有 observed duration 或任一选择
  分量非有限值时训练直接失败，不能生成指向不存在 best checkpoint 的完成摘要。

本次复审只运行 CPU 合成数据与静态 shell/contract 测试，没有连接远端、读取 sealed test
指标或启动 GPU。
