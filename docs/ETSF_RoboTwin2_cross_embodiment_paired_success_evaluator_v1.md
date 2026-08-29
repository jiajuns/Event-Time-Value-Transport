# ETSF RoboTwin2 跨本体配对成功率评估器 v1

## 1. 它回答什么问题

[`evaluate_robotwin2_cross_embodiment_paired_success_v1.py`](../scripts/evaluate_robotwin2_cross_embodiment_paired_success_v1.py)
把论文主张从 critic 诊断指标推进到标准的跨本体指标：在从训练与选择中留出的机器人本体上，冻结 actor
baseline 与 `ETSF best-of-4` 使用同一个 requested seed，比较完整任务成功率：

\[
\Delta SR = SR_{ETSF\ best\text{-}of\text{-}4} - SR_{actor}
\]

AUROC、Brier、时长 MAE 可以解释 critic 是否学到了信号，但不能替代这个主指标。阶段进度与事件链结构对齐，
但仍只是 supporting endpoint，不能把 full-task failure 改写成 success。

本评估器只消费已经冻结的静态 JSON；不会打开数据集、archive、trajectory、checkpoint、prediction tensor，
不会 reset simulator、查询 policy、训练、选择 checkpoint 或改变候选数。

## 2. 与五本体预注册的硬绑定

评估器只接受预注册 SHA：

`75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee`

冻结范围为：

- benchmark：`RoboTwin2.0`；
- task：`move_can_pot`；
- held-out bodies：`aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5`；
- evaluation conditions：`clean`、`randomized`；
- requested seeds：`2026090000..2026090099`；
- 每个 seed 的执行顺序按奇偶固定为 `actor_baseline → etsf_best_of_4` 或反序；
- 总计 `5 × 2 × 100 = 1000` 个 paired identities、`2000` 条 rollout。

输入必须与这个完整笛卡尔积完全相等。缺一行、增加一行、重复、换 seed、重试后替换、改变方法顺序都会失败，
因此 evaluator 不做 available-case 分析，也没有“看到结果后删掉失败 episode”的入口。

## 3. 唯一输入 schema

顶层字段必须恰好为：

```json
{
  "format": "etsf_robotwin2_move_can_pot_paired_outcomes_v1",
  "status": "frozen_complete_preregistered_five_body_two_condition_pairs",
  "preregistration_sha256": "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee",
  "rows": [],
  "rows_sha256": "canonical SHA-256 of rows",
  "document_sha256": "canonical SHA-256 of all preceding fields"
}
```

每一行字段也必须恰好为：

```json
{
  "benchmark": "RoboTwin2.0",
  "task": "move_can_pot",
  "heldout_body": "piper",
  "condition": "clean",
  "requested_seed": 2026090000,
  "method_order": ["actor_baseline", "etsf_best_of_4"],
  "actor_baseline_binary_success": 0,
  "actor_baseline_stage_progress": 0.5,
  "etsf_best_of_4_binary_success": 1,
  "etsf_best_of_4_stage_progress": 1.0
}
```

成功标签只接受 JSON integer `0/1`，不接受 boolean；进度只接受
`{0, 0.25, 0.5, 0.75, 1}`，且 progress 为 `1` 当且仅当对应 binary success 为 `1`。JSON
duplicate key、`NaN/Infinity`、未知字段、score/logit/probability/feature/path、训练或验证身份字段均被 exact schema
拒绝。输入文件还必须为内容 SHA 已知、只读、非 symlink 的 `.json` 普通文件。

注意：严格 schema 能证明“输入没有夹带预测信号且相对冻结 roster 没删行”，但仅凭一个结果文件不能证明它是在
看 outcome 之前预注册，也不能密码学证明真实训练集合和 held-out body 的互斥，更不能证明这些标签确实来自官方
simulator checker。这三项仍需独立、已签名的训练闭包与 rollout provenance verifier。

## 4. 报告层级

输出按预注册顺序包含：

1. 10 个 `body × condition` cell；
2. 5 个 body 的 equal-condition macro；
3. 2 个 condition 的 equal-body macro；
4. global equal-body-condition macro。

每个 cell 都恰好有相同的 100 个 requested seeds，所以在上述每层，equal-cell macro 的点估计与 pooled row
mean 数值相同；报告明确记录这一平衡设计条件，不把任意不平衡输入悄悄当 macro。

## 5. 成功率统计

每一层同时报告：

- actor baseline 与 ETSF best-of-4 的 success count 和 SR；
- 预注册主口径 Wilson 95% CI；
- 补充的 equal-tailed Clopper–Pearson 95% CI；
- paired `ΔSR`；
- 固定 20,000 次 requested-seed cluster percentile bootstrap 95% CI；
- 保守的 Bonferroni Clopper–Pearson marginal-difference CI（明确标为非 paired exact）；
- `n00`、actor-only success `b`、ETSF-only success `c`、`n11`；
- discordant pairs 上的 exact two-sided McNemar/binomial p-value。

bootstrap seed 固定为 `2026090200`，draw generator 固定为
`SHAKE256 uint64 rejection-modulo v1`，2,000,000 个冻结 draw index 的 SHA 固定为
`bcbc2e7c2f2761aca738ed7e2589e4cf9ffbc79460ff37ebb072b78077265149`；生成器或序列变化会失败。
重采样单位始终是 requested seed：

- 单 cell：一个 cluster 含 1 行；
- body macro：同 seed 的 clean/randomized 2 行一起抽；
- condition macro：同 seed 的 5 bodies 一起抽；
- global：同 seed 的 5 bodies × 2 conditions 共 10 行一起抽。

因此 global 不会错误地把 1000 行当成 1000 个独立 IID 单元。percentile bootstrap 不是 exact CI，报告字段名
明确包含 `not_exact`。

Wilson 和 Clopper–Pearson 的行级覆盖结论都没有处理 repeated-seed cluster；宏层存在 repeated-seed 依赖，
所以二者在宏层只是预注册描述/补充，不能替代 cluster bootstrap。类似地，代码按预注册计算 pooled discordant McNemar 的精确二项
p-value，但报告明确写出它没有建模宏层的 repeated-seed 依赖；不能把组合计算的 exact 偷换成 cluster-robust
推断。

## 6. 阶段进度统计

每层报告：

- 两种方法的 mean terminal max-event progress；
- paired progress delta；
- 与成功率相同的 requested-seed cluster percentile bootstrap CI；
- 以 100 个 requested-seed cluster mean 为独立单元、有限样本且通常较宽的 two-sided Hoeffding
  conservative CI（不把宏层的 200/500/1000 行当 IID）；
- 阶段阈值 `0/.25/.5/.75/1` 的 reach rate。

Hoeffding 和 bootstrap 都没有被称为 exact。阶段进度用于解释“卡在哪个事件”，不能挽救失败的 full-task
success gate。

## 7. 运行方式

先由外部、标签后置的结果物化器生成完整 JSON，冻结文件并计算字节 SHA：

```bash
chmod 0444 /absolute/non_sensitive/paired_outcomes.json
sha256sum /absolute/non_sensitive/paired_outcomes.json
python3 scripts/evaluate_robotwin2_cross_embodiment_paired_success_v1.py \
  --input /absolute/non_sensitive/paired_outcomes.json \
  --input-file-sha256 <上一步的64位SHA> \
  --output /absolute/non_sensitive/crossbody_report.json
```

输出使用 hard-link create-once 发布并冻结为 `0444`；已有 output、symlink output 或 symlink parent 均失败。
输出带 canonical `report_sha256`，并把 input file SHA、input document SHA、rows SHA 与 preregistration SHA
全部绑定起来。

## 8. 能力与论文主张边界

报告中的 training、collection、simulator/policy execution、action ranking、promotion、deployment、
cross-embodiment improvement claim capability 全部为 `false`。只有在外部 verifier 进一步证明以下事实后，真实
报告才可成为跨本体论文证据的一部分：

1. 每折 held-out body 在训练、adapter、calibration、route selection 中从未被使用；
2. actor/ETSF/checkpoint/candidate-N/tie rule/event-spec/runtime 在第一条 outcome 前冻结；
3. 1000 个 pair 全部来自相同 seed/reset/candidate contract，失败或 crash 没被重试替换；
4. binary success 来自官方 full-task checker，而不是 critic 预测或阶段进度；
5. 五个 LOBO fold 都完成，且 global 主 gate 与每个 body/condition 的方向性 gate 按预注册独立复核。

在真实 2000 rollouts 尚未执行和验证前，这个 evaluator 证明的是“统计实现已经完整、不会泄漏或删行”，不是
“ETSF 已经提高跨本体成功率”。
