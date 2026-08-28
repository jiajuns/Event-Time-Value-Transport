# Source dense260 v2 离线信任与预注册协议

实现入口为 `scripts/freeze_smolvla_schema5_source_dense260_protocol_v2.py`。它只处理 canonical JSON、公钥、SHA 和 detached signature；不构造环境、不 reset、不 step、不导入策略、不打开 HDF/trajectory/label，也不生成、读取或保管私钥。所有输出均以 `O_EXCL`、`0400` create-once 创建，并明确 `collection_authorized=false`。

## v2 修复的边界

v1 的 canonical SHA 只能证明内容地址，不能证明是谁签发。v2 增加一个需要外部 file SHA 与 logical SHA 双重固定的 public trust-root registry，并要求：

- candidate reset manifest 由专用 source-reset Ed25519 key 签发；
- official150、Source63、prior-development400、Piper development300、Formal190、evaluation400 六份 aggregate attestation 分别由 registry 授权的 reference custodian key 签发；
- source-reset key 与所有 reference key 必须不同；
- 每个 reference role 精确冻结 namespace、logical group count、membership semantics、原始 source format/namespace/count、source file/logical SHA 和 identity extractor SHA；
- 每份 attestation 绑定 registry SHA、candidate 文件/逻辑/source-authority SHA、reference source-contract SHA、三轴 count/commitment 和三轴 intersection=0。

registry 是 bootstrap trust root，不会自称第三方证明。部署审查者必须在运行任何后续命令前，从独立渠道固定 registry file SHA 和 `registry_sha256`。如果同一操作者控制两把 key，只能表述为 key-separated operator attestation。

## 精确角色合同

| role | count | 精确 membership |
|---|---:|---|
| official150 | 150 | RoboTwin `move_can_pot/success_seeds` 官方 150 |
| source63 | 63 | 已冻结 Schema5 Source63 requested/resolved/reset groups |
| prior_development | 400 | development150 后接、且与其不重叠的 V7 development250 union |
| piper_development300 | 300 | development300 identity-resolution receipt 的全部行 |
| formal_target_validation | 190 | 同一 development300 receipt 的 exact formal190 subset |
| evaluation400 | 400 | target seed manifest v2 的 exact evaluation lane |

Formal190 是 Piper development300 的显式子集。保留两份 attestation 是为了分别固定“整个 Piper development pool”和“正式验证 lane”的来源与签发权限，而不是声称两者彼此不重叠。

## Reset identity v2

v2 不接受 v1 identity。稳定 reset 行必须公开三个不可逆 SHA，并由 freezer 重算：

```text
reset_identity_sha256 = canonical_sha256({
  format: etsf_cross_body_semantic_reset_identity_v2,
  task: move_can_pot,
  instruction_semantics_receipt_sha256: ...,
  initial_scene_state_sha256: ...
})
```

`initial_scene_state_sha256` 的口径固定为现有 can/pot pose 的 float64 canonical JSON v1。requested/resolved seed、body、policy、joint 和 commanded drive state 均不进入 reset identity，因此同一语义场景不会只因本体不同而自动变成不重叠。candidate manifest 的 format、protocol namespace、identity contract 或任一可重算值仍为 v1 时直接失败。

## Registry 与外部签名

registry spec 只含 public `ssh-ed25519` keys，不接受 RSA/ECDSA、key options 或私钥。冻结命令要求显式提供并按字节 SHA 绑定 OpenSSH verifier：

```bash
python3 scripts/freeze_smolvla_schema5_source_dense260_protocol_v2.py \
  freeze-registry \
  --spec /immutable/dense260_v2/registry_spec.json \
  --ssh-keygen /usr/bin/ssh-keygen \
  --output /immutable/dense260_v2/issuer_registry.json
```

本协议没有 sign 子命令。持有者应在隔离签名环境中对 canonical JSON 文件使用预先存在的 key 和固定 namespace 生成 detached SSHSIG，例如：

```bash
ssh-keygen -Y sign \
  -f /offline/existing_ed25519_key \
  -n etsf-source-dense260-v2 \
  /immutable/dense260_v2/candidate_manifest.json
```

私钥路径不得传给本协议。freezer 通过以下等价命令语义验证，且 verifier 缺失、字节 SHA 漂移、超时、返回非零、principal/key/role 不匹配均 fail-closed：

```text
ssh-keygen -Y verify -f EPHEMERAL_ALLOWED_SIGNERS \
  -I FROZEN_PRINCIPAL -n etsf-source-dense260-v2 -s PAYLOAD.sig
```

## Aggregate attestation 与最终冻结

reference custodian 先在隔离位置准备一份 canonical、identity-only private view。它必须精确包含该 role 的 requested/resolved/reset identity 三轴，但不得包含 outcome、trajectory 或标签。协议验证其 source-lineage SHA、固定行数和零访问 capability 后，只发布 aggregate payload：

```bash
python3 scripts/freeze_smolvla_schema5_source_dense260_protocol_v2.py \
  prepare-attestation \
  --registry /immutable/dense260_v2/issuer_registry.json \
  --registry-file-sha256 REGISTRY_FILE_SHA \
  --registry-sha256 REGISTRY_LOGICAL_SHA \
  --candidate-manifest /immutable/dense260_v2/candidate_manifest.json \
  --candidate-file-sha256 CANDIDATE_FILE_SHA \
  --candidate-signature /immutable/dense260_v2/candidate_manifest.json.sig \
  --private-reference-view /isolated/identity_only_view.json \
  --reference-role official150 \
  --issuer-id reference-custodian-v1 \
  --output /immutable/dense260_v2/official150_attestation.json
```

输出 attestation 不含 identity 原值或 private source path，必须再由该 role 的 registry-authorized key 在外部签名。收齐六对 `payload.json + payload.json.sig` 后才能执行 `freeze`。最终 freezer 逐份复核 registry/candidate/source binding 和真实 Ed25519 signature，再按候选顺序取前 260 个 stable、resolved-unique reset；这 260 个 reset identity v2 必须同时唯一，否则不向后替换。随后按 canonical SHA 顺序固定 100/80/80，三轴 split overlap 必须分别为零，采集合同固定为 8 candidates、exec5、max200。

## 当前不能伪造的缺口

本实现不生成真实 registry、candidate manifest、private reference view 或签名。真实 source SHA 和 identity-only receipts 尚未齐备，尤其 evaluation400 target manifest、development300/Formal190 identity receipt，以及 official/Source63/prior-development 的统一 reset identity v2 都是外部 blocker。缺少任一输入时应停止，不得用 seed 相等、v1 hash、合成 commitment 或自哈希代替。

单元测试不生成私钥。它使用合法 Ed25519 public-key 编码和临时 verifier stub 验证调用、授权与失败路径，并使用系统 `/usr/bin/ssh-keygen` 验证 malformed SSHSIG 必须失败。生产 happy-path 必须由部署方使用预先存在的审核 key/signature 做一次独立集成验证。
