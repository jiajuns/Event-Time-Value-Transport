# Piper Schema6 多 seed 扩采集 v2：CPU 预注册协议

## 当前交付边界

本版本只交付可审计的 CPU 协议层：读取并验证未来冻结的
`etsf_smolvla_piper_target_seed_manifest_v2`，通过严格数据门选择
`adaptation=80` 与 `validation=50`，按 manifest 固定顺序生成 130 个逐 seed
命令契约。`production_execution_authorized=false`；脚本不会构造环境、reset、
加载或前向 policy、打开 HDF5、使用 GPU 或连接远端。`collect-one` 子命令当前
明确 fail-closed，不能误当成生产 collector。

目标 manifest JSON 本身是本协议唯一允许读取的 seed 输入，因此会验证其中
530 行身份元数据及全局唯一性；evaluation 的 400 行仅作为已签名 manifest
元数据完成结构验证。不会为 evaluation 生成命令、目录或 HDF5 路径，不会对
evaluation seed 构造环境、reset、执行 policy 或打开任何 episode/HDF5 数据。

## 为什么不能直接迁移 r6j 单 seed 脚本

现有 r6j 链有两个不满足多 seed 的硬约束：

- reset materializer、authority freezer 和 collection launcher 把
  `100101000` 固定在代码与 authority 中；其 object registry 只能证明该固定
  reset，不能复用于其他 seed。
- dense collector 只遍历 root 的合法候选。多 seed v2 要求对原始候选
  `[0,1,2,3]` 全部留下分支记录；不可行动作不能执行，但必须写为已签名的
  `nonexecuted/censored` 分支，才能保持每组精确四分支并避免候选选择偏差。

因此本次新增独立 v2 预注册器，不修改或冒充已冻结的 r6j 实现。r6j 仅作为
上游 runtime/code closure 被逐文件 SHA256 绑定。

## 冻结输入与硬门

预注册要求以下文件均为只读 regular file，并由调用方提供预期字节 SHA256：

- target seed manifest v2；同时验证其 `seed_manifest_sha256`；
- canonical `move_can_pot` event spec；链必须为
  `e0 -> e12 -> e3 -> e4 -> eK`，moving=`can`；
- r6j 七个 runtime/code artifact，closure 为
  `canonical_sha256({relative_name: file_sha256})`；
- 绑定的 Python 与未来 v2 runner。

任何输入、输出、lock 或嵌入路径只要 lexical/resolved component 含
`fresh` 或 `confirmation`，都会在文件访问前拒绝。输入不能是 HDF5。

数据门固定为：

- adaptation 精确 80，validation 精确 50，二者按 manifest 顺序串行；
- 530 个 requested/resolved/pair identity 在 manifest 中全局唯一；
- 仅 130 个 selected identity 被复制进命令；
- evaluation command/reset/policy/HDF5 计数均为 0；
- 顺序只由 split/ordinal 决定，success、reward、event 或 outcome 不能参与选择、
  排序或断点恢复。

## 每 seed 的生产实现义务

未来 v2 runner 必须在每个 seed 独立执行以下流程，不能接受固定 seed registry：

1. 用命令中的 requested seed reset，验证 resolved seed 和初始 scene SHA 与
   target manifest 一致。
2. 从该次 live task 重新构造 can/pot registry，核对 actor name、稳定 actor ID、
   model index 与 asset family（`105_sauce-can`、`060_kitchenpot`）。
3. 由该 registry 重新生成 pose spec，并验证 pose spec 绑定该 registry SHA。
4. 对候选原始索引 0、1、2、3 分别 reset；每次复验 seed identity、registry、
   pose、observation 与 root 一致。可行候选执行分支；不可行候选只写
   nonexecuted/censored 记录，禁止执行不安全动作。
5. create-once 写 group 与完成 receipt；完成 receipt 必须包含 group 文件 SHA、
   reset receipt SHA、registry SHA、pose spec SHA 和四分支计数。

## detached、4090 锁与恢复契约

预注册冻结如下生产要求，但当前 CPU 阶段不执行它们：

- 生产 watcher 必须 detached，并留下 detached launch receipt；
- 整个 reset/collection 写入窗口必须持有指定 lock 文件的独占 `flock`，并再次
  验证实际设备为 RTX 4090；
- output root 与每组输出均 create-once，不能覆盖 partial 或旧文件；
- 恢复只能接受从第 0 组开始、连续无空洞的已签名
  `etsf_smolvla_piper_schema6_multiseed_group_receipt_v2` 前缀；
- 每个 receipt 必须与 preregistration SHA、command SHA、seed identity 精确一致，
  且生产 watcher 要重新计算 group 文件字节 SHA；伪造、跳号、partial 或未签名
  结果一律 fail-closed。

`validate_completed_prefix()` 已实现上述 receipt 结构与无空洞前缀验证。真正的
detached launcher、GPU lock 持有、per-seed simulator runner 尚未授权或实现，
不得据此文档声称已开始远端扩采集。

## CPU 测试覆盖

定向测试覆盖：精确 80+50 顺序、四候选命令、evaluation 零命令/零 HDF 访问、
逐 seed reset receipt 路径唯一、manifest/SHA 变更拒绝、敏感路径前置拒绝、
create-once、production 子命令 fail-closed，以及 signed gap-free resume prefix。
