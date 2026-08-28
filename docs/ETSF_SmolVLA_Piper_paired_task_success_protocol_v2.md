# SmolVLA/Piper evaluation400 成对成功率协议 v2

## 边界

`scripts/smolvla_piper_paired_success_protocol_v2.py` 是新的、独立的协议冻结层，
不修改历史 paired v1。它只认证 JSON 元数据，并把五个 target-adapter checkpoint
作为不透明字节计算 SHA；不反序列化模型，不导入策略，不调用仿真器，也不打开
HDF5、轨迹、预测、validation label、evaluation outcome 或 test 文件。

协议唯一允许的最终成对 lane 是 target manifest 中已有的 400 个 evaluation identity。
每个 identity 在同一初态上执行一次 baseline 和一次 ETSF，顺序来自 identity bridge v2
中冻结的 `condition_order`。不创建、也不要求第二套 reserve400。

## 四个硬依赖

冻结器必须同时精确绑定：

1. `etsf_smolvla_piper_evaluation400_paired_identity_bridge_v2`；
2. 独立签发的 `etsf_smolvla_piper_evaluation400_paired_execution_authority_v2`；
3. calibrator 的 `etsf_smolvla_piper_multitask_head_support_v2`，六个头
   `post_event/next_event/duration/success/recovery/object_effect` 均须进入 primary，且
   recovery 必须满足 support、五成员均已训练并完成 validation-only temperature；
4. r7h source lineage 派生的五成员
   `etsf_smolvla_piper_adapter_ensemble_manifest_v2` 及五份
   `etsf_smolvla_piper_schema6_adapter_member_receipt_v2`。

单 checkpoint 与任何 LOBO checkpoint 都不被接受。五个成员的 seed、checkpoint
path/SHA、training manifest SHA、split SHA、prediction contract SHA 和 r7h source
ensemble contract SHA 必须逐项一致且互异。checkpoint 只做普通文件 SHA，不加载 header。

## 独立 execution authority v2

仓库原有 identity bridge 只声明所需 authority format，并没有实现该 authority。
因此新增 `scripts/freeze_smolvla_piper_evaluation400_execution_authority_v2.py`。
它要求一个由独立执行方预先签发的
`etsf_smolvla_piper_evaluation400_independent_execution_decision_v2`，并重新物化、逐项比较
identity bridge 及其七个依赖。decision 必须在任何 outcome/trajectory 被读取前签发，
明确授权同一 evaluation400、禁止额外 reserve、禁止冻结后修改 seed/candidate/threshold，
且只授权外部 executor。协议冻结器本身仍然没有执行权限。

## 结果协议（本轮不执行）

外部 executor 后续必须产生恰好 400 条、无重复且顺序一致的 pair result。每条结果绑定
`pair_id`、实际 condition order、同一 reset identity 的复核、baseline/ETSF 二元成功值，
以及两条独立 execution receipt SHA。缺失、重复、身份不符或非二元成功值均失败关闭。

最终 evaluation receipt 预注册以下统计量：

- 成功率差：`mean(etsf_success - baseline_success)`，报告 95% CI；
- McNemar：由 `n00/n01/n10/n11` 构成，对 discordant `n01/n10` 做双侧精确二项检验；
- paired bootstrap：以 `pair_id` 为抽样单位、有放回、固定 seed `20261103`、20,000 次，
  对成功率差给出双侧 percentile 95% CI。

不得事后筛 seed、candidate、threshold 或 subgroup。

## 冻结顺序

先由独立方生成并签名 decision JSON，再冻结 execution authority：

```bash
python3 scripts/freeze_smolvla_piper_evaluation400_execution_authority_v2.py \
  --identity-bridge /immutable/evaluation400_identity_bridge_v2.json \
  --identity-bridge-file-sha256 "$BRIDGE_FILE_SHA" \
  --external-decision /immutable/independent_execution_decision_v2.json \
  --external-decision-file-sha256 "$DECISION_FILE_SHA" \
  --expected-r7h-source-ensemble-contract-sha256 "$R7H_SOURCE_CONTRACT_SHA" \
  --output /new_root/evaluation400_execution_authority_v2.json
```

再冻结 paired protocol：

```bash
python3 scripts/smolvla_piper_paired_success_protocol_v2.py \
  --identity-bridge /immutable/evaluation400_identity_bridge_v2.json \
  --identity-bridge-file-sha256 "$BRIDGE_FILE_SHA" \
  --external-authority /new_root/evaluation400_execution_authority_v2.json \
  --external-authority-file-sha256 "$AUTHORITY_FILE_SHA" \
  --head-support /immutable/head_support_v2.json \
  --head-support-file-sha256 "$HEAD_SUPPORT_FILE_SHA" \
  --ensemble-manifest /immutable/target_adapter_ensemble_v2.json \
  --ensemble-manifest-file-sha256 "$ENSEMBLE_FILE_SHA" \
  --adapter-member-receipt /immutable/member_0_receipt.json "$M0_RECEIPT_FILE_SHA" \
  --adapter-member-receipt /immutable/member_1_receipt.json "$M1_RECEIPT_FILE_SHA" \
  --adapter-member-receipt /immutable/member_2_receipt.json "$M2_RECEIPT_FILE_SHA" \
  --adapter-member-receipt /immutable/member_3_receipt.json "$M3_RECEIPT_FILE_SHA" \
  --adapter-member-receipt /immutable/member_4_receipt.json "$M4_RECEIPT_FILE_SHA" \
  --expected-r7h-source-ensemble-contract-sha256 "$R7H_SOURCE_CONTRACT_SHA" \
  --output /new_root/paired_success_protocol_v2.json
```

两个输出都使用 create-once hard-link 发布并设为 owner-read-only `0400`。直接输入路径若
包含 fresh、confirmation、test、trajectory、label，或是 symlink/HDF，会在读取前失败。

## 尚未授权的外部工作

本实现不含 simulator executor，也不含读取 400 对真实结果并计算统计量的 evaluator。
后续必须另建独立外部 executor/result evaluator，并由其验证 protocol file SHA、每个 reset
identity、两条 condition execution receipt 和完整 400 对之后，才能生成结果 receipt。
