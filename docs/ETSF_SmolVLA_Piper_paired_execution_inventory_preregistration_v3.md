# Paired v3 production execution inventory/key preregistration

## 目的与生产边界

`scripts/preregister_smolvla_piper_paired_execution_inventory_v3.py` 独立生成或接收三类
Ed25519 key，并冻结 paired-success v3 所需的 trusted issuer attestation 与 execution
inventory。它不导入、修改或解除
`smolvla_piper_paired_success_protocol_v3.APPROVED_EXECUTION_INVENTORY_FILE_SHA256`。

只有以下实现全部最终冻结、逐项独立复审并取得 expected file SHA 后，才可运行 production
`preregister`：

- external executor；
- final result evaluator；
- simulator runtime adapter；
- collector condition runner；
- runtime contract；
- container inventory。

本工具产出 inventory 不等于 production allowlist 已批准，更不授权 evaluation400 执行。独立
审计方仍需审核最终 inventory 文件，之后在单独变更中固定其 file SHA。当前 paired v3 allowlist
保持 fail-closed。

## Key 身份

工具接受三个不同的 32-byte raw Ed25519 public key：

- issuer key：写入 trusted issuer attestation；
- executor key：其原始公钥 SHA256 精确写入 `executor.identity_sha256`；
- result-signer key：其原始公钥 SHA256 精确写入
  `result_evaluator.identity_sha256`。

因此 executor/result identity 不是自由填写的字符串。结果评估器以后验证 result receipt 时，
必须拿 result-signer public key 的 raw bytes 重新计算同一 SHA。public key manifest 用于安全分发
公钥，inventory 只保存这些不可混淆的 identity SHA。

## 可选 key 生成

```bash
python3 scripts/preregister_smolvla_piper_paired_execution_inventory_v3.py \
  generate-keys \
  --output-directory /absolute/new/key_root
```

该子命令 create-once 创建：

- `issuer_ed25519_private.raw`、`executor_ed25519_private.raw`、
  `result_signer_ed25519_private.raw`：32-byte raw private key，mode `0400`；
- 三份对应 raw public key：mode `0444`；
- `public_key_manifest.json`：mode `0444`，仅包含 public key 信息，不公开 private key 路径或
  private key SHA。

输出目录最终为 mode `0500`。私钥不得进入代码仓、inventory、日志或命令行。已有受管 key
可以直接跳过 `generate-keys`，将其 raw public key 传给 `preregister`。

## Inventory 冻结

所有 path 必须是 canonical absolute path，并同时提供外部 expected file SHA：

```bash
python3 scripts/preregister_smolvla_piper_paired_execution_inventory_v3.py \
  preregister \
  --issuer-key-id external-evaluation-issuer-v3 \
  --issuer-public-key /keys/issuer_ed25519_public.raw \
  --issuer-public-key-file-sha256 "$ISSUER_PUBLIC_FILE_SHA" \
  --executor-public-key /keys/executor_ed25519_public.raw \
  --executor-public-key-file-sha256 "$EXECUTOR_PUBLIC_FILE_SHA" \
  --result-signer-public-key /keys/result_signer_ed25519_public.raw \
  --result-signer-public-key-file-sha256 "$RESULT_PUBLIC_FILE_SHA" \
  --executor-implementation /final/paired_executor_v3.py \
  --executor-implementation-file-sha256 "$EXECUTOR_IMPL_SHA" \
  --result-evaluator-implementation /final/paired_result_evaluator_v3.py \
  --result-evaluator-implementation-file-sha256 "$RESULT_EVALUATOR_IMPL_SHA" \
  --simulator-implementation /final/smolvla_piper_schema6_runtime_adapter_v2.py \
  --simulator-implementation-file-sha256 "$SIMULATOR_ADAPTER_SHA" \
  --collector-implementation /final/condition_runner.py \
  --collector-implementation-file-sha256 "$COLLECTOR_RUNNER_SHA" \
  --runtime-contract /final/runtime_contract.json \
  --runtime-contract-file-sha256 "$RUNTIME_CONTRACT_SHA" \
  --container-inventory /final/container_inventory.json \
  --container-inventory-file-sha256 "$CONTAINER_INVENTORY_SHA" \
  --condition-runner-binding distinct \
  --issuer-attestation-output /new/authority/trusted_issuer_attestation.json \
  --inventory-output /new/authority/execution_inventory_attestation.json
```

生产执行必须使用 `distinct`：`simulator_implementation` 绑定实际构造 RoboTwin/SmolVLA
runtime 的 `smolvla_piper_schema6_runtime_adapter_v2.py`，`collector_implementation` 绑定最终
`execute-condition-v3` condition runner。runner 必须按 supervisor 传入的 path/SHA 动态加载并
复核该 simulator adapter，不能把自身同时声明成 simulator 而把真实 runtime adapter 留在未绑定的
传递依赖里。

工具仍支持 `shared`，用于两个角色确实由同一份实现承担的非生产合同测试；此时两个 canonical
path/SHA 必须完全一致。`shared` 不是当前 Piper evaluation400 的生产推荐。声明与实际 descriptor
不符会失败关闭。

## Exact paired-v3 contract

issuer attestation 和 inventory 的 format/status/字段集合与
`smolvla_piper_paired_success_protocol_v3._validate_execution_inventory` 完全一致。合成测试会临时
注入 synthetic inventory SHA 后直接调用该验证器；生产代码没有测试注入或 allowlist 写操作。

inventory 固定：

- 唯一 evaluation400 lane，`pair_count=400`、reserve count `0`；
- executor/result evaluator 的实现 path/SHA 与公钥派生 identity；
- simulator/runtime/collector/container 的 exact path/SHA；
- `component_inventory_complete=true`；
- 两个真实实现存在；
- attestation 期间 outcome/trajectory 文件打开数严格为 integer `0`。

## Fail-closed 读取与输出

所有输入逐层使用 `openat`/`O_NOFOLLOW`，同一 FD 完成读取和 SHA，读前后核对
device/inode/size/mtime；symlink、可写组件、非普通文件、HDF、敏感 namespace、SHA 漂移均拒绝。
runtime contract 和 container inventory 还必须是 strict JSON object；duplicate key、NaN、Infinity
失败关闭。所有 count 使用 `type(value) is int`，所以 `false` 不能冒充 `0`。

两个最终 JSON 都以 `O_EXCL` create-once 创建并冻结为 mode `0444`。本流程打开 HDF、trajectory、
label、outcome 的数量始终为 `0`，也不执行 simulator、policy、executor 或 evaluator。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_preregister_smolvla_piper_paired_execution_inventory_v3.py
```

测试只使用临时 synthetic key/脚本/JSON，不连接远端，不打开真实数据或 checkpoint。
