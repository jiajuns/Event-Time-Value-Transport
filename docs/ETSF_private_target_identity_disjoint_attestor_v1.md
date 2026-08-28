# ETSF 私有目标身份不相交 attestor v1

该工具只在持有保留身份的隔离本地进程内比较集合，发布物不包含任何身份值或私有输入路径。它不会访问环境、策略、轨迹、标签或网络。当前仓库只用合成集合测试过它；没有据此声称真实目标集合已经完成认证。

实现入口：`scripts/attest_private_target_identity_disjoint_v1.py`。

## 私有输入契约

所有私有输入必须是当前用户持有的普通文件、权限精确为 `0400`、非符号链接。JSON 必须没有多余字段。

单集合格式（`role` 分别为 `heldout` 或 `candidate_pool`）：

```json
{"format":"etsf_private_identity_set_v1","status":"private_local_attestation_material_only","role":"heldout","identities":[1,2]}
```

已选择 requested/resolved 格式：

```json
{"format":"etsf_private_identity_set_v1","status":"private_local_attestation_material_only","role":"selected_requested_and_resolved","requested":[11,12],"resolved":[21,22]}
```

身份必须是非负整数；每个列表非空、内部唯一。requested 与 resolved 长度必须相等。私有文件本身禁止复制到公开 artifact 目录。

## 三个阶段

先在隔离位置生成 heldout 集合承诺：

```bash
chmod 0400 /PRIVATE/heldout.json
python scripts/attest_private_target_identity_disjoint_v1.py commit-heldout \
  --private-heldout /PRIVATE/heldout.json \
  --output /PUBLIC/heldout-commitment.json
```

候选池认证要求调用方同时给出 target plan 已绑定的候选列表 SHA：

```bash
chmod 0400 /PRIVATE/candidate-pool.json
python scripts/attest_private_target_identity_disjoint_v1.py attest-candidate-pool \
  --private-heldout /PRIVATE/heldout.json \
  --heldout-commitment /PUBLIC/heldout-commitment.json \
  --private-candidate-pool /PRIVATE/candidate-pool.json \
  --expected-target-identity-set-sha256 PLAN_BOUND_SHA \
  --output /PUBLIC/candidate-disjoint-attestation.json
```

reset-only 选择完成后，再对 requested+resolved 联合身份承诺认证：

```bash
chmod 0400 /PRIVATE/selected-requested-resolved.json
python scripts/attest_private_target_identity_disjoint_v1.py attest-selected-requested-resolved \
  --private-heldout /PRIVATE/heldout.json \
  --heldout-commitment /PUBLIC/heldout-commitment.json \
  --private-selected /PRIVATE/selected-requested-resolved.json \
  --expected-target-identity-set-sha256 RESET_RECEIPT_BOUND_SHA \
  --output /PUBLIC/selected-disjoint-attestation.json
```

输出路径必须尚不存在；工具以 `0400` 创建，拒绝覆盖。任何 schema、权限、承诺、集合交集或目标 SHA 不一致都会在创建输出之前失败，错误文本不回显身份、SHA 或文件路径。

## SHA 与下游兼容性

- heldout：`canonical_sha256(sorted(identities))`，因此同一集合不受输入顺序影响；承诺只发布 count、SHA、状态和自签名。
- candidate pool：`canonical_sha256(candidates)`，与 target plan 的有序候选池承诺一致。
- selected：`canonical_sha256({"requested": requested, "resolved": resolved})`，与 reset receipt 的 `identity_set_sha256` 一致。
- 两种不相交认证均是 `etsf_private_identity_disjoint_attestation_v1`，精确通过 `smolvla_piper_target_seed_manifest.validate_disjoint_attestation` 的闭合字段校验；除该校验强制的格式/角色/状态字段外，只发布 SHA、`intersection_count=0`、不含敏感身份声明和签名。

这只是隐私最小化的本地承诺协议，不是带密钥的第三方密码学证明；可信边界仍包括运行该命令的隔离操作者和本地进程。
