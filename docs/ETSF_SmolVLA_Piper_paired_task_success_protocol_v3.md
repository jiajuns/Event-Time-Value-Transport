# SmolVLA/Piper paired-success protocol v3

## 目的与边界

`scripts/smolvla_piper_paired_success_protocol_v3.py` 是独立的新协议层，不修改 paired
v1/v2，也不执行 simulator、policy 或结果统计。它把授权拆成两个阶段：

1. `freeze-core` 冻结完整的 pre-outcome protocol core，且
   `execution_authorized=false`；
2. 独立评测方对该 core 的 digest 做 Ed25519 签名；`freeze-bundle` 重建 core、验证
   签名并发布只供外部 executor 使用的 bundle。

因此 external decision 绑定的是稳定的 `protocol_core_sha256`，而 bundle 只是 core 与
decision 的内容寻址清单，不需要反向进入 decision，避免了协议 SHA 与 authority SHA 的
循环依赖。

canonical SHA 只证明内容完整性，不被当成签名。若 `cryptography` 的 Ed25519 实现不可用，
流程失败关闭，不允许退回 SHA “签名”。

## core 输入闭包

core 要求调用方分别提供路径和外部 expected file SHA：

- post-collection v3 static plan、terminal receipt、identity-bridge handoff；
- 五份 `etsf_smolvla_piper_schema6_adapter_member_receipt_v3`；
- evaluation400 identity bridge v2；
- reviewed post-v3 launcher SHA；
- 一个 independently reviewed、源码中 file-SHA allowlist 锁定的
  `execution_inventory_attestation`。它必须完整列出真实 executor、真实 result evaluator、
  simulator adapter、runtime contract、collector adapter、container inventory，以及可信
  issuer allowlist attestation。

issuer key、issuer identity、executor identity 和 result-evaluator identity 只能来自该冻结
inventory，不能由 `freeze-core` CLI 自由指定。当前真实 executor/result evaluator 尚未完成独立
审计，因此源码中的 allowlist 常量故意为 `None`，生产 `freeze-core` 会明确失败关闭；合成测试
仅用测试内 monkeypatch 验证协议路径，不能产生可部署 bundle。待实现和完整 execution inventory
审计完成后，必须把最终 inventory **文件 SHA** 硬编码进新版本源码再解除该门。

post-collection v3 launcher 的独立审计最终 SHA 已固定为
`7c46cea4677dea23c1fd7b50aa9da3af6999880b6c5196bb995dd36fcbfb67c4`；协议要求源码常量、
CLI `--expected-post-launcher-sha256`、post static plan 中的 implementation SHA、以及实际
launcher 文件 SHA 四者完全一致。

static plan 还必须精确包含七个 implementation role（含 `r9b_watcher`）以及
`python_import_closure`。协议逐项验证闭包 module 名、canonical path、file SHA 和实际不透明
文件字节，并要求每个 implementation 的精确 path/SHA 记录都出现在闭包中；缺少
`r9b_watcher`、闭包条目、传递模块文件或任一 SHA 变化都会失败关闭。

验证器从这些根递归重建并交叉核对：

- post-v3 artifact closure 和 formal190 global one-shot claim；
- development300 的 train80/internal30/formal190 分区；
- formal190 evaluator authority/terminal receipt；
- formal190 calibrator input authority、calibration、head support、ensemble 和 terminal
  receipt；
- 五个 native r7h source member 与五个 Piper target adapter 的一一对应关系；
- 六个 primary heads，包括 conditional recovery 的训练、支持和 validation-only
  temperature；
- policy feature/runtime/action mapping；
- target manifest 中唯一的 evaluation400 lane、400 个 pair identity 和冻结顺序。

单 checkpoint、LOBO checkpoint、aggregate checkpoint、joint teacher 和第二套
reserve400 均不接受。abstention threshold 在 core 中转换为固定点
`{coefficient, decimal_places}`，不把浮点作为新的可调参数。

## 安全读取

所有输入通过逐层 `openat`、`O_NOFOLLOW` 和单一文件描述符读取：同一批字节同时用于文件
SHA 和 JSON 解析，并在读取前后用 `fstat` 比较 device/inode/size/mtime，避免“先 hash 再
重新打开”或读中替换的 TOCTOU 窗口。任何祖先 symlink、末端
symlink、非普通文件、可写冻结 artifact、HDF 路径或敏感 namespace 都失败关闭。

严格 JSON parser 拒绝 duplicate key、NaN 和 Infinity。所有计数使用
`type(value) is int`，因此 `true` 不能冒充 `1`。member 的 prediction/label NPZ 路径只作为
已签名 metadata 保存，不会被本协议打开；十个 source/adapter checkpoint 只作为不透明
字节计算 SHA，不做 header 或模型反序列化。

## Ed25519 decision

外部签名方构造的 exact statement 包含：

- core file SHA 和 `protocol_core_sha256`；
- issuer key/public-key/identity 和固定 executor identity；
- execution inventory 的 file/logical SHA、完整执行栈 binding、executor/result-evaluator
  implementation SHA 与两个固定 identity；
- 32-byte nonce、`authorization_sequence=1`；
- pair identity set、deployment binding、policy/runtime/action binding；
- exact 400、唯一 evaluation lane、无 reserve400；
- 禁止冻结后修改 seed/candidate/threshold/order；
- decision 前未读取 outcome/trajectory；
- `external_executor_only=true`。

签名消息为：

```text
b"ETSF/SmolVLA/Piper/paired-v3/execution-authority\0"
+ canonical_json(statement)
```

decision envelope 的 `decision_sha256` 仍只是内容摘要。真实性只来自
`decision_signature_ed25519_hex` 与 core 预绑定 public key 的验证。

## 统计预注册

core 固定：

- 恰好 400 条完整 pair result；
- 二元 success；
- `mean(etsf_success-baseline_success)`；
- discordant `n01/n10` 的 exact two-sided McNemar；
- 以 pair ID 为抽样单位、固定 seed `20261103`、20,000 次的 paired percentile
  bootstrap 95% CI；
- 禁止事后 seed、candidate、threshold、subgroup 选择。

本实现不读取结果，也不实现真实 executor/result evaluator。

## CLI

阶段一：

```bash
python3 scripts/smolvla_piper_paired_success_protocol_v3.py freeze-core \
  --post-plan /immutable/post_v3/_watcher/static_plan.json \
  --post-plan-file-sha256 "$POST_PLAN_FILE_SHA" \
  --post-terminal /immutable/post_v3/final_receipt.json \
  --post-terminal-file-sha256 "$POST_TERMINAL_FILE_SHA" \
  --post-handoff /immutable/post_v3/handoff/evaluation400_identity_bridge_v2_handoff.json \
  --post-handoff-file-sha256 "$POST_HANDOFF_FILE_SHA" \
  --member-receipt /immutable/post_v3/members/member_0/final_receipt.json "$M0_FILE_SHA" \
  --member-receipt /immutable/post_v3/members/member_1/final_receipt.json "$M1_FILE_SHA" \
  --member-receipt /immutable/post_v3/members/member_2/final_receipt.json "$M2_FILE_SHA" \
  --member-receipt /immutable/post_v3/members/member_3/final_receipt.json "$M3_FILE_SHA" \
  --member-receipt /immutable/post_v3/members/member_4/final_receipt.json "$M4_FILE_SHA" \
  --identity-bridge /immutable/evaluation400_identity_bridge_v2.json \
  --identity-bridge-file-sha256 "$BRIDGE_FILE_SHA" \
  --execution-inventory-attestation /immutable/execution_inventory_attestation_v3.json \
  --execution-inventory-attestation-file-sha256 "$INVENTORY_FILE_SHA" \
  --execution-inventory-attestation-sha256 "$INVENTORY_LOGICAL_SHA" \
  --expected-post-launcher-sha256 "$REVIEWED_POST_V3_LAUNCHER_SHA" \
  --runtime-execution-authority /immutable/schema6_execution_authority.json \
  --runtime-execution-authority-file-sha256 "$RUNTIME_AUTHORITY_FILE_SHA" \
  --selector-implementation /immutable/select_smolvla_piper_evaluation400_root_candidate_v3.py \
  --selector-implementation-file-sha256 "$SELECTOR_FILE_SHA" \
  --output /new_root/paired_protocol_core_v3.json
```

独立签名方离线产生 decision JSON 后，阶段二：

```bash
python3 scripts/smolvla_piper_paired_success_protocol_v3.py freeze-bundle \
  --core /new_root/paired_protocol_core_v3.json \
  --core-file-sha256 "$CORE_FILE_SHA" \
  --decision /external/paired_execution_decision_v3.json \
  --decision-file-sha256 "$DECISION_FILE_SHA" \
  --output /new_root/paired_execution_bundle_v3.json
```

core 和 bundle 都是 create-once、owner-read-only `0400`。

core 中的 `selector_authority` 逐项绑定 formal190 calibration/root-ranker、五成员 adapter checkpoint、每成员 source composite rank contract、object normalization、共享 deployment uncertainty 实现以及上面的 runtime execution authority 文件 SHA。root primary score 是 source composite group-relative margin；六个 factual prediction head 只进入已验证预测、不确定性和消融。

为让 external executor 与 result evaluator 不依赖 proof 自报，`core.deployment`、
`core.deployment.selector_authority` 和 `core.development_and_formal190` 三处保存并逐值比较同一
冻结闭包：

- `source_rank_score_contracts`：按五成员顺序的五个完整、内容寻址 contract；
- `deployment_parameters`：exact keys 为 `post_event_temperature`、
  `next_event_temperature`、`success_temperature`、
  `conditional_recovery_temperature`、`duration_scale_multiplier`、
  `object_scale_multiplier`、`object_error_robust_scale_m`、
  `deployment_uncertainty_contract_sha256`；
- `formal190_thresholds`：exact keys 为 `minimum_formal190_composite_margin`、
  `maximum_formal190_pair_uncertainty`、`maximum_global_total_uncertainty`、
  `root_group_ranker_sha256`。

paired freezer 会把这些值与重新打开并验证的 calibration、ensemble、五份 member receipt 和
identity bridge 逐项核对；不允许 executor/result evaluator 另报一套参数。

runtime authority 不是不透明 JSON。协议解析其带逻辑 SHA 的 nested `runtime_contract`，要求其
SHA 精确等于 identity bridge 的 `target_reset_runtime_contract_sha256`，并要求
`max_episode_steps` 为严格整数 `200`。core 中只发布 exact
`{path,file_sha256,nested_runtime_contract_sha256,max_episode_steps}` 记录，同时镜像 target-reset
SHA。paired freezer 还会重新打开 bridge 绑定的 target manifest，并核对 bridge 顶层 target-reset
SHA 确实来自该 manifest；`true`、`200.0`、不同 nested SHA、伪造 bridge mirror 或不同步数都会
失败关闭。

## 尚未实现

真实 simulator executor、evaluation400 一次性消费账本、每条件 execution receipt 和最终
结果 evaluator 均保持外部缺口；在 executor/result-evaluator inventory 的最终审计 SHA 写入
allowlist 前，`freeze-core` 本身也不可用。本协议不会将“验证签名成功”误报为“已经执行评测”。
