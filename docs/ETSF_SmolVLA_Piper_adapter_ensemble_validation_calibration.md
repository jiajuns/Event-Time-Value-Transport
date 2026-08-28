# SmolVLA/Piper 五成员 adapter validation-only ensemble calibration

## 用途与边界

`scripts/calibrate_smolvla_piper_adapter_ensemble.py` 在五个 Piper adapter 训练成员完成后，使用冻结的 validation predictions/labels 完成 ensemble 指标、不确定性分解、success/conditional-recovery temperature 与 abstain threshold 冻结。它不修改 trainer，不加载 checkpoint 参数，只计算 checkpoint 文件 SHA；不会读取 test、Fresh、confirmation、paired-development outcome 或 HDF5。

正式输入必须由五成员冻结后的独立 target-validation evaluator 产生，不能用 trainer 的 internal validation 代替；后者无法满足至少 50 个 validation groups 的 abstention 门。历史 v2 lane 是 50 组，development300 lane 是 190 组；两者均由 authority 精确绑定，且 evaluation400 不进入评估权限。

该产物是进入 baseline↔事件插件成对任务成功率实验之前的 calibration/dependency 层，不是任务成功率结果。validation 指标不能用于宣称跨本体成功率提高。

## 输入 authority

入口只接受一个带逻辑 SHA 的 JSON：

- `format=etsf_smolvla_piper_adapter_ensemble_validation_input_v2`；
- `status=frozen_five_member_validation_predictions_before_calibration`；
- `lane=validation_only`；
- 恰好五个成员，`member_index=0..4` 且 seed 唯一；
- 五个成员必须逐字共享 `training_manifest_sha256`、`split_sha256` 和 `source_ensemble_contract_sha256`；
- 每个成员绑定 checkpoint 路径/文件 SHA 和 validation prediction NPZ 路径/文件 SHA；checkpoint 和 prediction SHA 在五成员间必须唯一；
- 一个共享 validation label NPZ，绑定文件 SHA 与按顺序 sample identity-set SHA；
- `test_artifacts_read=false`、`fresh_artifacts_read=false`、`confirmation_artifacts_read=false`。

任何路径 component 命中被禁用的数据命名空间、任何 symlink、HDF 后缀、SHA 漂移、成员 sample 顺序变化、pickle/object array 或形状变化都会 fail closed。输出 root 必须不存在。

prediction NPZ 的严格字段为：

```text
sample_id
post_event_logits [N,C]
next_event_logits [N,C]
success_logit [N]
source_contract_base_rank_score [N]
source_action_rank_residual [N]
source_contract_rank_score [N]
recovery_logit [N]
duration_log_mean [N]
duration_log_scale [N]
object_mean [N,O]
object_log_scale [N,O]
```

label NPZ 的严格字段为：

```text
sample_id [N]
group_id [N]
group_row_ordinal [N]
current_event [N]
post_event [N]
next_event [N]
success [N]
regress [N]
recovery [N]
recovery_observed [N]
duration [N]
duration_observed [N]
object_target [N,O]
object_observed [N]
root_candidate [N]
candidate_index [N]
is_baseline [N]
candidate_final_success [N]
```

## 指标与不确定性

post/next event 先逐成员 softmax，再平均概率，报告 ensemble NLL、macro-F1、accuracy 和 10-bin equal-width ECE。不确定性使用精确 entropy 分解：

```text
total predictive entropy
= mean member entropy          # aleatoric
+ ensemble mutual information # epistemic
```

success temperature 在 validation 上用固定 `[0.05,20]` 对数网格最小化五成员平均概率的 binary NLL，随后报告 NLL、Brier 和 ECE。其 law-of-total-variance 为：

```text
total Bernoulli variance
= mean[p_m(1-p_m)] # aleatoric
+ Var[p_m]         # epistemic
```

recovery 使用相同的五成员 Bernoulli 温度与方差分解，但只允许 `recovery_observed & regress` 行参与拟合和质量统计；未观察到恢复的右删失 regress 不能伪装成负例。五个 checkpoint 必须共同证明 detached recovery head 已训练，且 validation 中 recovery 正负各至少 10 个独立组，否则温度固定为 `1.0`、该 head 不进入主效用或结构化不确定性。

duration 与 trainer 的监督精确对齐：模型输出的是 `log(1+D)` 的五成员等权 shifted-lognormal mixture，通过 mixture CDF 二分求出 `D=exp(Y)-1` 的 median 与 central 90% interval，在 observed validation rows 上报告 median MAE 和 coverage。原始时间尺度上的方差分解为成员 shifted-lognormal variance 均值加成员物理时间均值方差。

object head 使用五个 diagonal Gaussian 成员。ensemble mean 为成员均值，total variance 为成员方差均值加成员均值方差；报告每维 aleatoric/epistemic/total variance、central 90% marginal coverage 和 joint coverage。

## 五折 group cross-fit 性能门

support 只是必要条件。六头的温度或 scale 先按完整 `logical_group_id` 做固定五折 cross-fit，OOF 指标按 logical group 等权；post/next 对 persistence/prior，success/recovery 对 prevalence，duration 对 current-event group median，object 对 zero/robust-median，并同时要求固定 group-bootstrap 的零增益 LCB 与 uncertainty/AURC 门。长轨迹不会获得更大权重。OOF 仅决定性能门；过门后才在全部 formal190 上拟合 deployment temperature/scale，二者在 artifact 中明确分离。

## Head support 门

每个 head 在 receipt 中记录并由 paired validator 复核自己的冻结 minimum；core prediction head 使用 10 groups/side，稀疏且可能直接影响排序的 optional head 保持 50 groups/side：

- post/next：每个输出 event class 的最小 group support，minimum=10；
- duration：observed/censored groups，minimum=10；
- success：positive/negative groups，minimum=50；
- conditional recovery：仅 observed operational-regress 行的 positive/negative groups，minimum=10，并要求五成员 recovery head 全部已训练；
- object effect：nonzero/near-zero effect groups，minimum=50。

支持不足的 head 仍可输出明确标注的 descriptive 数值，但不能进入 primary structured uncertainty/utility。success 支持不足时禁止 temperature 拟合，deployment temperature 固定为 `1.0`。输出的 `paired_head_support.json` 使用现有 paired protocol 所要求的 `etsf_smolvla_piper_multitask_head_support_v1` 格式；如果 post/next 支持不足，paired protocol 会继续 fail closed。

## Abstain threshold

共享纯函数 `smolvla_piper_deployment_uncertainty_v1.py` 使用 full-refit deployment 参数复算线上不确定性；duration/object multiplier 在调用前各应用恰好一次，object 必须已经反归一化为 physical xyz metre。initial pre-action e0 root 的唯一 policy 是 `excluded_at_initial_e0_without_observed_operational_regress`：root gate 固定平均 post/next/success/duration/object 五个适用头，recovery 保留为验证预测与消融，但不混入 e0 root 分母。共享实现 path/file SHA、policy、head count=5 和 object robust scale 全部写入 calibration、root-ranker、ensemble 和 paired selector authority。

阈值候选固定为 validation uncertainty 的 `0.50/0.60/0.70/0.80/0.90/0.95/1.00` quantile。每个候选以完整 `group_id` 为 bootstrap unit，固定 seed `20260828`、生产默认 5,000 次重采样，计算 retained multi-head quality 的 95% LCB。候选必须同时满足：

- retained groups 至少 50；
- retained sample coverage 至少 50%；
- quality LCB 至少 0.60。

在合格候选中选覆盖率最高者。没有候选满足时发布 `disabled_no_threshold_meets_validation_lcb` 和 `maximum_total_uncertainty=0`，不得从 test 或后续 paired outcome 回头调阈值。

formal190 root ranker 与 factual success calibration 严格分离：排序只用每成员 `source_contract_rank_score=source_contract_base_rank_score+source_action_rank_residual` 的同组 lowest-legal-baseline 相对 margin，绝不把它解释成 success logit/probability。候选变化还必须同时通过全局 candidate uncertainty 和 candidate/baseline pair uncertainty；margin 使用严格 `>`，平分回退 baseline。固定门为 changed>=50、discordant>=20、每折 changed>=10/discordant>=4、paired gain bootstrap LCB严格大于0、harmful-rate UCB不超过0.10。

## 输出与 paired 兼容性

成功运行会 create-once 写出并冻结只读：

- `calibration.json`：指标、分解、support enablement 和 bootstrap 阈值；
- `paired_head_support.json`：paired protocol 可直接绑定的 head support；
- `ensemble_manifest.json`：五个 checkpoint、temperature、enablement 和 threshold 的 deployment contract；
- `final_receipt.json`：`format=etsf_smolvla_piper_adapter_ensemble_validation_receipt_v2`、`status=complete_validation_only_five_member_calibration`，带 `receipt_sha256`；
- `run.exit`：严格内容 `0\n`。

因此 paired dependency authority 可以将此 `final_receipt.json` 作为 `adapter` dependency，绑定 format/status/逻辑 SHA、`validation_only=true`、`test_hdf5_files_opened=0`、所需防泄漏字段和根 `run.exit`。真正启动 paired development 前还必须要求 `abstain_threshold_enabled=true`；该字段为 false 时只能保留 calibration 审计，不能绕过阈值门。paired protocol 的 `adapter_checkpoint_path` 可以绑定 `ensemble_manifest.json`；运行时 selector 的 `adapter_checkpoint_sha256` 应解释为该 ensemble manifest 的文件 SHA，而不是任一单成员 checkpoint。

## 本地 CPU 测试与运行

```bash
cd /home/jj/Event-Time-Value-Transport
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_calibrate_smolvla_piper_adapter_ensemble.py
```

测试使用纯合成五成员 NPZ，覆盖多头指标、uncertainty identity、support 禁用、group bootstrap LCB、敏感路径拒绝、只读收据以及现有 paired dependency validator 兼容性。合成数值不是模型效果。

生产命令只能在五个真实 adapter 成员及其 validation authority 冻结后执行：

```bash
python3 scripts/calibrate_smolvla_piper_adapter_ensemble.py \
  --input-authority /IMMUTABLE/validation_input_authority.json \
  --input-authority-file-sha256 INPUT_FILE_SHA256 \
  --output-root /ABSENT/adapter_ensemble_validation_output
```

本次交付没有连接远端，也没有执行此生产命令。
