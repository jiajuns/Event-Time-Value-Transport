# ETSF 五折 OOF 结构化预测诊断

## 为什么需要独立诊断

`oof_selection.json` 的问题是“候选重排是否在 development OOF 上改善成功率”，不是
“世界模型的每个预测头是否准确”。成功率 guard 通过不能代替事件、时间和对象变化预测精度；
反过来，某个预测头指标较好也不能放宽 guard。本诊断因此写入独立文件
`oof_prediction_diagnostics.json`，不进入 scoring grid、guard grid 或 fresh50 授权条件。

所有样本都来自五折各自的 held-out groups：每个 logical group 只能由唯一 owner fold 产生
一次预测。评估器只读五个 `oof_predictions.pt`，不接收数据集目录或 fresh seed manifest，
所以不会读取 fresh50。输出中的
`fresh_confirmation_data_or_labels_read=false` 和 `authorization_guard_changed=false` 是协议断言，
不是通过预测精度检验的标志。

## 指标

- 成功概率：报告未校准 ensemble，以及严格 cross-fit temperature 后的 Brier、binary NLL、
  ECE、accuracy、ROC-AUC 和 PR-AUC/AP。评估 fold 的 temperature 只由其余四个 OOF folds
  拟合；同时报告由其余四折 prevalence 构成的常数 baseline，以及按 logical group 聚类的
  paired Brier/NLL bootstrap CI。组内 success-changing pairs 另外报告 ties=0.5 的排序准确率和
  相对随机 0.5 的组级 CI。
- 下一事件：在 structured mask 上报告 top-1 accuracy、multiclass NLL、各类 support/recall；
  同时报告 class precision/F1 和 present-class macro-F1；
  对比“下一事件等于当前事件”的 persistence baseline，以及只由其他 folds 标签构成的平滑
  class-prior baseline。另报告 observed destination event 和 post-predicate 的概率指标。
- 持续时间：训练目标是 `log1p(D)` 上的 Normal，因此 observed 样本报告 ensemble-mixture
  NLL、steps MAE/RMSE 和 log-MAE；right-censored 样本使用 `P(D > censor_bound)` 的 survival
  NLL，绝不把 censor bound 当成精确持续时间。`reach_logit` 另外按 observed/censored 二分类
  报告 Brier/NLL/AUC。observed point error 与只使用其他四折 observed 标签拟合的
  `clock_event_id × body_id` median baseline 比较；无精确 key 支持时按 event、body、global
  顺序回退，回退次数写入产物。
- 对象状态变化：先用 factual checkpoint 的冻结 mean/std 反归一化到物理位置 delta，再报告
  coordinate MAE/RMSE、平均 L2 error、Gaussian-mixture NLL，并与 zero-delta baseline 比较。
- 失败/恢复：failure/success 在 terminal branches 上评估。只有 checkpoint 明确记录
  `recovery_supervised=true` 时才评估三分类 recovery；否则输出
  `not_evaluable_model_contract_recovery_supervised_false`，同时保留 recovery/regression 标签支持数，
  不把未训练的第三类伪装成准确率。
- 不确定性：成功、下一事件、observed duration 和对象变化分别报告 uncertainty-error Spearman、
  低/高不确定性四分位误差和 selective-risk AURC；分类任务还报告用不确定性检测错误的 ROC-AUC。
  每项 AURC 同时报随机排序的期望风险（即总体 mean error）、相对随机改善及五折逐折结果。
  这些量说明不确定性能否排序风险，不等价于成功率提升。

成功、事件、时长、对象和 outcome 都包含每折 held-out 指标或 support；pooled 指标只是五个互斥
held-out partitions 的合并，不是训练内拟合结果。

## raw artifact 兼容性

新的 schema-v5 fold artifact 在每个 row 增加可选
`structured_predictions.format=etsf_oof_structured_prediction_row_v1`，包含每个 ensemble member 的
事件 logits、duration distribution、reach、object distribution、outcome 和对应 mask/label。原有
`member_success_logits/member_event_progress/member_normalized_duration/member_aleatoric` 字段及 reduction
语义不变，所以 frozen guard 决策不变。

旧 train100 artifact 没有这些字段时仍能重新计算成功概率诊断，但结构化部分必须显示
`legacy_raw_schema_partial/not_evaluable`；禁止从候选级 event progress 反推虚假的 next-event accuracy。
同一次 OOF 中新旧 row 混用会 fail closed。

development250 的第五候选 `sample_blend_1.000` 只用于训练和附录诊断。主成功校准、组内排序和
prediction adequacy 固定只使用 fresh/部署共同拥有的前四候选；第五候选无论预测多好都不能抬高
部署准确性结论。

## 与 reranking guard 独立的预测充分性门

`prediction_adequacy.protocol=etsf_development_prediction_adequacy_v1` 在查看 development250 结果前
固定，且不被任何模型超参、scoring grid 或 guard threshold 使用。其通用阈值为：

- paired loss 使用 equal-logical-group estimand、固定 seed `20260903`、10000 次 group bootstrap；
  model-minus-baseline 的双侧 95% CI 上界必须严格小于 0；
- 二分类每类至少 10 个样本；事件至少 2 个 present classes，且每个 present class 至少 5 个；
- success 的 10-bin equal-width ECE 必须不高于 `0.10`；
- 每个 post-event predicate 的正类与负类各至少 10 个，Brier 的组级 CI 必须优于只由其他四折
  拟合的 Beta(1,1)-smoothed prevalence，且 macro-F1 必须严格优于该 baseline；任一谓词不满足
  即 fail closed，不能用 micro accuracy 中的大量负类掩盖；
- duration/object 比较至少覆盖 30 个 logical groups；
- success PR-AUC 必须严格高于 prevalence，event/outcome macro-F1 必须严格高于相应 baseline；
- success within-group pair error 必须以组级 CI 严格优于随机 0.5；
- 成功、事件、时长、对象四种不确定性均须 Spearman 为正、总体 AURC 优于随机，且 5/5 折
  的 AURC 均优于随机。五折全胜在随机胜负概率 0.5 的单侧 fold-sign null 下为 `1/32`。

只有 success、next-event、duration、object、failure/success（有监督时包括 recovery）及
uncertainty 全部通过，`generic_development_predictive_skill_pass` 才为 true。任何缺失支持或失败
均 fail closed。该门只许可“当前 development 分布上相对冻结 baseline 的通用预测 skill”这一表述；
其 `fresh50_authorization_effect=none`，不能授权候选 guard、fresh50、绝对任务安全、跨策略或跨本体。

## 结论边界

该文件是 development held-out prediction evidence。只有独立 prediction adequacy 通过，才能写
“在当前 OpenVLA/RoboTwin development 分布上相对冻结 baseline 有预测能力”。即使通过，也没有
预注册的米制对象容差、时间 deadline 或代价敏感成败阈值，因此不能简写为“绝对准确”。它仍不能
单独证明跨策略、跨本体迁移，也不能替代 guard-authorized 后唯一一次 fresh50 成功率确认。
