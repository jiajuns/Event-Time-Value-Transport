# Schema6 training manifest v2：label-blind 聚合协议

## 结果与边界

`materialize_smolvla_piper_schema6_training_manifest_v2.py` 将多个未来冻结的
Piper Schema6 collection root 聚合成训练输入。它只读取签名 JSON、object
registry 与 pose-quality spec，不调用 `h5py`，不读取 HDF5 字节、dataset、
success、reward、event 或其他标签。

HDF5 在聚合阶段是 opaque sealed artifact。聚合器只做：非 symlink regular
file、只读、非空、root 内相对路径，以及 authority/manifest/final receipt 中
记录 SHA256 的逐字段一致性验证。HDF 字节 SHA 将由 trainer 在外部 split 已
冻结后、且仅对允许打开的 train/validation 组重新计算；聚合器自身绝不通过
打开 HDF 来验证字节。

## 外部身份门

输入 target manifest 必须是冻结的
`etsf_smolvla_piper_target_seed_manifest_v2`，调用方同时提供文件 SHA 与逻辑
`seed_manifest_sha256`。聚合器先验证全部 identity metadata，再冻结唯一允许的：

- adaptation：80 组；
- target validation：50 组；
- evaluation：0 组。

collection root 的顺序不影响输出。每个 root 通过 requested/resolved seed、
split、ordinal 与 pair ID 匹配 target manifest；重复、未知或 evaluation identity
均 fail-closed。所有选择和排序只依赖已冻结 identity，不依赖 outcome。

## 每个 collection root 的精确合同

root 必须整体只读并包含：

- `collection_authority.json`
- `manifest.json`
- `final_receipt.json`
- `object_registry.json`
- `pose_quality_spec.json`
- `schema6_group.hdf5`

三个合同分别使用：

- `etsf_smolvla_piper_schema6_collection_authority_v2`
- `etsf_smolvla_piper_schema6_collection_manifest_v2`
- `etsf_smolvla_piper_schema6_collection_final_receipt_v2`

不允许任何未声明字段，以防 collection metadata 携带标签。聚合器验证：

- 三者 identity、四候选 `[0,1,2,3]` 与精确四分支一致；
- target manifest 文件/逻辑 SHA、event spec SHA、collector lineage SHA；
- authority、manifest 的文件 SHA 与逻辑 SHA；
- HDF recorded SHA 在 manifest/final receipt 中一致；
- registry 精确为 can、pot，验证 live actor ID 格式、asset family/model ID、role；
- pose spec 合法且逻辑 SHA 绑定该 seed root 的 registry SHA；
- `per_seed_live_registry_materialized=true` 且
  `fixed_seed_registry_reused=false`；
- Fresh/confirmation、evaluation、real-robot 与性能声明权限全部关闭。

路径在任何 filesystem 访问前进行 lexical/resolved component 检查；包含
`fresh` 或 `confirmation` 的路径一律拒绝。

## 不足与充分输出

不足 130 个目标 identity 时，只 create-once 写：

- `schema6_training_manifest_v2_receipt.json`

其状态为 `insufficient_data_no_training_authorized`，包含已到达数量、缺失的外部
identity、target partition SHA 和 expected split SHA；不生成训练 manifest。

130 组全部到达时，create-once 写：

- `schema6_target_partition_v2.json`
- `schema6_external_group_split_v2.json`
- `schema6_training_manifest_v2_compat.json`
- `schema6_expected_manifest_split_v2.json`
- `schema6_training_manifest_v2_receipt.json`

compat manifest 使用当前 trainer 可由 `scan_manifest()` 读取的 v1 schema，包含
130 个 group 的相对路径与 recorded HDF SHA。expected receipt 同时冻结 manifest
文件/逻辑 SHA、target partition 文件/逻辑 SHA、external split 文件/逻辑 SHA。

## 60/20/50 外部 split

为满足正式支持门且保护 target validation：

- adaptation80 由 `sha256(1701:logical_group_id)` label-blind 排序；前 20 为
  internal validation，其余 60 为 train；
- target validation50 全部为 sealed test；
- evaluation400 完全不进入任何输出。

聚合器的 sufficient expected receipt 还绑定更新后的 formal trainer 实现 path/file SHA，
并记录 `direct_bound_trainer_execution_authorized=true`。该 trainer 强制接收 expected
receipt 路径与文件 SHA，在任何 HDF 字节哈希或 `h5py.File` 前复验自身实现、compat
manifest、target partition、external split 的文件/逻辑 SHA，以及 60/20/50 精确互斥
全覆盖；随后直接使用 external train60/internal-validation20。formal CLI 不再接受 split
seed/fraction 重分。target validation50 全部为 sealed test，其 HDF 永不打开或字节哈希。
trainer path/SHA、expected receipt 或任一 membership 不一致都会 fail-closed。

Phase2 生产根的权威形式是签名 preregistration 加每组 `per_seed_reset_receipt.json` / `completed_group_receipt.json`。聚合器可直接验证这条原生 lineage，并只对 HDF 做 `lstat` 和跨 receipt 的记录 SHA 一致性检查，不读取 HDF 字节。由于 terminal collection 整树会冻结只读，trainer manifest 对已认证 group 使用绝对路径；这些路径仍受 manifest、external split、expected receipt 和 trainer implementation SHA 的联合绑定，不能用软链或未签名路径替换。

## 测试

CPU synthetic 测试使用并非合法 HDF5 的 opaque bytes，并拦截所有 `.hdf5`
打开操作，覆盖：

- signed insufficient receipt；
- 130 组充分聚合与当前 trainer `scan_manifest()` 兼容；
- bound trainer 自身 SHA 与 expected receipt 授权；
- 外部 60/20/50 与 target validation 全量 sealed-test；
- expected receipt 错误文件 SHA 拒绝；
- evaluation root 拒绝；
- manifest/final HDF recorded SHA 不一致拒绝；
- can asset registry 错误拒绝；
- 敏感路径在 metadata 打开前拒绝。
