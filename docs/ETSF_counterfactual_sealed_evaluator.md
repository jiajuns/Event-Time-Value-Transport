# Schema-v5 反事实 ensemble 一次性 sealed 评估

入口：`scripts/evaluate_openvla_etsf_counterfactual_sealed.py`。

该命令只接受冻结的 `ensemble_manifest.json`、显式 schema-v5 sealed candidate 根目录、训练时 event spec 和新输出目录。CLI 没有 temperature、score weight、guard threshold 或覆盖参数；所有决策常数均从 ensemble manifest 与内嵌 checkpoint 的一致契约加载。

```bash
python scripts/evaluate_openvla_etsf_counterfactual_sealed.py \
  --ensemble-manifest /path/to/frozen/ensemble_manifest.json \
  --sealed-data /path/to/schema_v5_sealed_candidates \
  --event-spec /path/to/event_spec.json \
  --output /path/to/new_sealed_evaluation \
  --device cpu
```

严格确认实验还必须追加冻结的 reset-only fresh-50 manifest：

```bash
  --fresh-seed-manifest /path/to/fresh_confirmation_preregistered_resolved.json
```

只有 collection manifest 同时满足 `seed_registry=explicit_fresh_confirmation`、
`fresh_seed_manifest_sha256` 匹配、ordered requested/resolved 50 seeds 与冻结 manifest
完全一致时，输出才标为 `fresh_confirmatory`。原官方 test 即使成功完成评估，也固定标为
`development_holdout`；seed `100100000` 的协议事件使其不能再恢复为 untouched confirmation。

正式运行前应冻结 ensemble manifest、aggregate/member checkpoint、event spec 与输出目录。ensemble/event-spec/fresh-manifest/aggregate/member 路径均禁止位于 sealed 根目录内，避免恶意或误配的 provenance 在预约前别名到 sealed 文件。命令会在读取 sealed collection manifest、计算 sealed HDF5 hash 或打开标签 dataset 之前，以 `O_EXCL` 创建 `evaluated_once.json` 占位；已有完成结果或中断占位均拒绝重跑，也没有 `--overwrite`。成功后同一路径被原子替换为最终结果。

审计包括：

- manifest/aggregate/member checkpoint SHA256；
- event-spec SHA256；
- 训练、验证与 sealed logical key/解析 seed 无重叠；
- fresh-50 registry、冻结 seed manifest SHA 及 ordered requested/resolved seed；
- 显式 sealed 根目录与训练产物记录的 `sealed_test_groups/files` 完全一致；
- schema-v5、语言、候选顺序、deterministic baseline、动作干预、完整轨迹和 continuation-query 契约；
- HDF5 SHA 与训练前 identity-only sealed provenance 一致；
- 离线评分逐项复算 temperature、event、duration、candidate-distance 与 uncertainty 公式，并检查数值漂移。

主结果是 frozen guard 后的选择；unguarded top-1 仅作诊断。输出包含 actor/guarded/unguarded/oracle 成功率、paired delta 与固定种子 bootstrap 95% CI、改善/伤害/候选改变数、exact sign/McNemar p、candidate score/probability AUC、组内 pair accuracy、guard coverage、fallback 原因直方图和逐组审计记录。

同一次、同一冻结模型还会输出独立的 `prediction_metrics`，分别统计首动作候选和包含 continuation 的全部查询：post-event/relative-transition/next-reached-event 的 support-aware macro-F1、balanced accuracy、confusion matrix、概率 mixture NLL 与 self-loop/frequency 基线；每个动态谓词及 reach/success 的 ROC-AUC、PR-AUC、F1、mixture log-loss、Brier、ECE；包含右删失的 duration log-time mixture NLL 与 observed median-prediction MAE；对象 xyz 位移 raw-meter 误差及 normalized diagonal-Gaussian joint-mixture NLL；每成员在自身 encoder 坐标内的 future-latent cosine/NLL；以及 success uncertainty 的 risk-coverage/AURC。离散概率先逐成员归一化再平均；duration 和对象先形成每成员 likelihood、再以 log-sum-exp 做等权 mixture。future latent 不跨不同成员的潜在坐标直接混合。

mask 固定为：event/relative/predicate 仅用 `structured_mask`，reach/duration/object/future 仅用 `dense_mask`，next-reached-event 还要求 duration observed，success/outcome/AURC 只用首候选的 terminal 标签，continuation 的 success/outcome 占位符永不计入。AUC 使用 rank 公式保持线性内存；AURC 对同 uncertainty 的样本按组内随机顺序期望处理，避免由 HDF5 行顺序产生虚假优势。这些指标只做预测审计，不参与阈值、评分权重或 guard 的选择。

当前入口评估离线同状态候选分支，不等同于重新执行闭环在线 rollout。

正式 fresh-50 不应手工拼接这些命令；使用
`scripts/launch_openvla_etsf_fresh50_confirmation.py`，由其在消费 fresh 前验证冻结
validation guard/evidence 门、执行 label-free collection readiness，并保证一次性评估后
才启动 progress baseline。详见 `docs/ETSF_fresh50_confirmation_launcher.md`。

evaluator 同时识别旧 validation-split contract 和 OOF authorized final contract。OOF
分支在创建 `evaluated_once.json` 预约前完成 selection/final manifest/fold artifacts SHA、
authorization、enabled guard 与 one-shot fresh-only 契约校验；OOF final 不允许用于官方
development holdout，也不允许缺少 `--fresh-seed-manifest`。
