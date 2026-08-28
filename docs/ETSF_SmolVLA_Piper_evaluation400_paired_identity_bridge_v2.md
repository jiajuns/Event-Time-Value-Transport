# SmolVLA/Piper evaluation400 → paired-success 身份桥 v2

## 设计结论

`scripts/freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2.py` 将 target seed
manifest 的 `splits.evaluation` 400 组定义为唯一最终 baseline–ETSF 成对执行 lane。不再创建、
请求或假定额外 reserve400：

```text
target manifest evaluation400
        ↓ 只打开 reset identity JSON，不打开 outcome/trajectory
paired identity bridge v2（400 个原 pair_id）
        ↓ 仍需独立外部 execution authority v2
baseline vs ETSF paired execution
```

原 paired-success v1、冻结的 Schema6 v2、target seed manifest v2 均不修改。v2 桥是新的
create-once 身份/部署绑定协议，不把旧 development lane 或额外 reserve 投影成最终评估数据。

## 冻结前置条件

桥接器只接受 `.json` 普通文件，每个输入都必须带调用者预先给出的外部文件 SHA-256。它在生成
任何 pair authority 前逐项验证：

1. 完整 target seed manifest v2：严格为 adaptation80、validation50、evaluation400；530 个
   requested/resolved seed 和 `pair_id` 唯一，evaluation 行的 stage role 必须是
   `sealed_paired_evaluation`；manifest 只做过 reset，step/policy/outcome 计数均为零。
2. 独立 selected-identity disjoint attestation：逻辑签名、全 530 组 identity-set SHA、heldout
   commitment 和 manifest 内嵌 attestation SHA 必须逐字一致，intersection 必须为零。
3. 五成员 deployment：ensemble manifest 必须精确包含 5 个不同 checkpoint SHA；核心
   `post_event/next_event/duration` head 已通过 validation-only 支持门。
4. calibration、head-support 与 abstention：三者的文件 SHA/逻辑 SHA 必须被 calibration terminal
   receipt 共同绑定；abstention 必须是 validation-group bootstrap LCB 冻结结果，至少保留 50 个
   validation group，且没有读取 test/paired outcome。
5. SmolVLA policy bridge verification：必须是结构化 reversible-event interface，绑定精确的
   checkpoint bridge contract SHA、960D runtime binding SHA、state-feature SHA 和显式 action
   mapping SHA。仅维度相同或自报 mapping 不构成通过。

桥内同时冻结 Formal190 selector 的完整、值级可复核 authority。它按五个 member 的固定顺序
携带五份带 `contract_sha256` 的 `source_rank_score_contracts`，并携带以下两个 exact mapping：

- `deployment_parameters`：四个 temperature、duration/object deployment scale、object error
  robust scale，以及 deployment uncertainty contract SHA；
- `formal190_thresholds`：composite margin、pair uncertainty、global total uncertainty 与 root
  group-ranker SHA。

这些值必须分别等于 bridge deployment mirror；selector 的 calibration/root-ranker/uncertainty
SHA 也必须与 deployment 一致。数值必须 finite，`bool` 不能冒充数值。它们均来自已验证的
calibration/ensemble 输入，而不是由 paired executor 或结果 evaluator 自报。

这里的 calibration terminal receipt 同时承担 abstention 冻结收据：它绑定签名 calibration
artifact；v2 桥进一步把完整 abstain-threshold 对象规范哈希为
`abstention_contract_sha256`，避免只绑定一个布尔值。

## 400 对身份与顺序

每个输出 pair 原样继承 target manifest evaluation 行的规范 `pair_id`，以及 requested/resolved
seed、instruction SHA、scene/measured-joint/commanded-target 三个初态 SHA。桥接器不重新发明第二套
pair identity。

每对都要求 baseline 和 ETSF 在各自条件执行前重新 reset 并复核完全相同的初态哈希；两者共享
同一根观测、四个根动作候选和相同 continuation。baseline 固定为 lowest-legal 候选，ETSF 固定为
五成员事件世界模型 + uncertainty abstention，弃权时回退 baseline。

条件顺序使用固定 namespace 的 `SHA256(namespace:pair_id)` 首字节最低位决定。它只依赖已冻结
身份，与成功、事件、时长或轨迹无关。400 行顺序严格等于 target manifest evaluation ordinal
`0..399`（对应 global ordinal `130..529`）；打开结果后不能重排、删 pair 或改变条件顺序。

## 权限上限

桥接脚本自身只发布 identity-only pair authority，并明确记录：

- HDF5、trajectory、evaluation outcome、label、checkpoint 打开数均为 0；
- environment reset/step、policy import/forward、pair condition 执行数均为 0；
- 不授权性能、迁移或成功率提升声明；
- `execution_authorized_by_this_bridge=false`。

执行端仍必须取得独立的
`etsf_smolvla_piper_evaluation400_paired_execution_authority_v2`。外部 authority 必须绑定 bridge
文件 SHA、逻辑 `bridge_sha256`、`pair_identity_set_sha256`，并在执行前重新哈希所有 dependency；
缺少任一绑定都不能打开 evaluation outcome/trajectory 或执行策略。

## 生成命令

以下只展示接口；本轮没有对任何真实服务器或封存文件运行：

```bash
python3 scripts/freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2.py \
  --target-manifest /ABS/target_seed_manifest.json \
  --target-manifest-file-sha256 TARGET_MANIFEST_FILE_SHA256 \
  --selected-identity-attestation /ABS/identity_attestation.json \
  --selected-identity-attestation-file-sha256 IDENTITY_ATTESTATION_FILE_SHA256 \
  --ensemble-manifest /ABS/ensemble_manifest.json \
  --ensemble-manifest-file-sha256 ENSEMBLE_MANIFEST_FILE_SHA256 \
  --calibration /ABS/calibration.json \
  --calibration-file-sha256 CALIBRATION_FILE_SHA256 \
  --head-support /ABS/paired_head_support.json \
  --head-support-file-sha256 HEAD_SUPPORT_FILE_SHA256 \
  --calibration-receipt /ABS/calibration_final_receipt.json \
  --calibration-receipt-file-sha256 CALIBRATION_RECEIPT_FILE_SHA256 \
  --policy-bridge-receipt /ABS/policy_bridge_verification.json \
  --policy-bridge-receipt-file-sha256 POLICY_BRIDGE_RECEIPT_FILE_SHA256 \
  --output /ABS/absent/paired_identity_bridge_v2.json
```

输出使用 `O_EXCL` 等价的硬链接发布语义，create-once、owner-read-only `0400`，包含规范 JSON
`bridge_sha256`。本地测试只构造合成 JSON，不包含任何真实 evaluation/test/trajectory/label/HDF。

## 纯 CPU 合成测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2.py
```

测试覆盖 400 个原生 `pair_id` 的完整顺序、无额外 reserve、selected-identity attestation、五成员
与核心 head/abstention 门、完整 source-rank/selector 参数与阈值传播、selector/deployment
值级不一致、bool-as-number、runtime/action SHA 防篡改、文件 SHA 防漂移、桥内顺序防篡改、
HDF 输入拒绝和 create-once `0400` 输出。
