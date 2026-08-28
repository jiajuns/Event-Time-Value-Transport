# Schema6 Piper target development 300 组标签盲预注册

## 边界与目的

`scripts/preregister_smolvla_piper_schema6_target_development300.py` 是一个只使用
Python 标准库的纯 CPU 预注册器。它不接受任何输入文件，不读取 seed registry、reset 状态、
trajectory、HDF5、成功/失败标签或任何 fresh、confirmation、evaluation/test 数据。它也不执行
仿真、真实机器人或策略推理。

该文件建立的是一个新的 create-once 协议，不修改、替代或声称兼容已冻结的 v2 协议。其目的
是在采集和标签打开之前，冻结 300 个 target development 组的 seed 候选、互斥成员关系、支持
配额及非自适应停止规则。

## 物理互斥划分

300 个组（每组固定 4 个候选分支，共计划 1200 个分支）一次性、标签盲地按确定性 SHA-256
排序冻结为：

| 物理 split | 组数 | 用途 |
|---|---:|---|
| adaptation train | 80 | adapter 训练；不得接触 formal validation |
| adaptation internal validation | 30 | adaptation 内部选择/支持审计；不并入 train |
| formal target-validation | 190 | 五个冻结 adapter 之后的独立支持审计与校准；不得训练或选 checkpoint |
| 合计 | 300 | target development 总量 |

因此 adaptation bucket 是 `80+30=110`，formal target-validation 是 190。另有 400 组
evaluation 继续完全封存：不属于这 300 组、不读取其 identity/membership、也不计入样本量。

这也修正了旧样本量口径：现有 v2 的 adaptation80 已经包含 train60/internal20，故其物理最低量
是 `60+20+50=130`，不是 150。新的 300 组是 `80+30+190`，三个 split 均互不重叠。

## Seed 候选与执行授权

预注册器从独立的数值 seed namespace 构造 300 个唯一候选，并用 namespace 与 seed 的
SHA-256 顺序做标签盲分配；已知的早期单组 seed `100101000` 被排除。每个逻辑组都绑定 task、
body、policy、actor、split、全局/局部序号、requested seed 与 identity SHA。

本预注册器不能验证候选 seed 是否在实际环境中 bit-exact resolve，也不能在不读取封存身份的
前提下独自证明与 evaluation400 的 disjointness。因此输出状态明确为
`collection_not_authorized`。开始正式采集前，必须由新的 collection/materialization 合同提供：

1. 每个 requested seed 的 exact resolved-seed 收据；不匹配即 fail-closed；
2. 外部、标签盲、只披露“不相交”结论的 evaluation-disjointness 收据；
3. create-once collection authority，且不得改写本预注册文件或替换失败 seed。

## 配额不是分组依据

事件、duration 和 recovery 的真实频率无法由 Source63 的 34/218 汇总识别，因此它们只作为
采集全部 300 组之后的 head 激活门：

- train/internal：outcome 正/负/discordant 各至少 5 个独立组；每个 post-event 与
  observed-next-event 类至少 5 行；observed/censored duration 各至少 5 行；conditional
  recovery 正负各至少 10 个独立组，right-censored non-recovery 不能当负例；
- formal target-validation：success 正负各至少 50 个独立组；每个 event 类至少 10 个独立组；
  observed/censored duration 各至少 10 个独立组；abstention/LCB 至少保留 50 个组；
- versioned calibrator v2 已实现 conditional recovery 校准，但本预注册本身仍不授权激活。独立
  validation authority 必须证明五个 recovery head 均已训练、正负各至少 10 个独立 regress 组，
  且 right-censored non-recovery 未被当作负例；任一条件不满足时保持 off-primary。

不得为了满足配额查看封存标签后换 seed、移动 split 或早停。固定的停止规则是收齐全部 300 个
预注册组的终态；某个 head 支持不足时只关闭该 head 并发布 insufficient-support 收据。追加组
必须建立新的 create-once 预注册，不能事后扩写本协议。

adaptation 标签也只能在所有 300 个 membership 与 collection artifact 冻结后打开，且只能打开
train/internal。formal target-validation 标签需要五个 adapter 均冻结后的单独 authority；本
预注册本身不授权打开这些标签。

## 生成与校验

```bash
python3 scripts/preregister_smolvla_piper_schema6_target_development300.py \
  --output /absolute/non_sensitive_path/schema6_development300_preregistration.json
```

输出是 create-once、`0444`、规范 JSON SHA-256 签名的不可变预注册。脚本在写入前重建整个
确定性文档并核对签名、300 个唯一组、三个 split 的完整覆盖与无重叠。命令只生成协议，不授权
采集、训练、校准或性能/跨本体迁移声明。

## Trainer v3 交接合同

完成采集后，新的物化流程必须生成 trainer 可验证的 v3 authority；不能把 300 个 ID 列表直接
作为 CLI 参数交给 trainer，也不能让 trainer 根据列表长度猜测 profile。v3 的三个签名 JSON
格式固定为：

- expected receipt：`etsf_smolvla_piper_schema6_expected_manifest_split_v3`；
- target partition：`etsf_smolvla_piper_schema6_target_partition_v3`；
- external split：`etsf_smolvla_piper_schema6_external_group_split_v3`。

三者都必须显式签名绑定 `split_profile=development300_v3`。expected receipt 与 external split
还必须分别携带完全相同的
`required_trainer_group_counts={train:80, validation:30, test:190}`；partition 必须携带
`required_group_counts={adaptation:110, formal_target_validation:190}`。profile、格式、任一计数、
成员覆盖、partition SHA、split SHA、manifest SHA 或 trainer 实现 SHA 不一致时均 fail-closed。
历史 v2 仍按其已发布的签名 `...v2` 格式身份和显式 `60/20/50` expected-receipt 计数验证，
不会由 130 个列表长度推断出来，也不会被静默升级为 v3。

trainer 的唯一 HDF 读取边界只遍历 `train` 和 adaptation-derived `validation`。`test` 中的
190 个 formal target-validation ID 不被传给读取函数；训练 summary 和 internal-validation
artifact 都记录 `development300_v3`、`80/30/190`、sealed group count 190，以及五个 adapter
checkpoint 全部冻结前 formal target-validation 的 HDF/label 打开数均为 0。之后是否释放这
190 组，只能由独立的五 adapter 冻结后 authority 决定。

## 2026-08-28 远端冻结实例

纯 CPU 预注册已在 4090 服务器上以 create-once 方式生成；该动作没有读取任何输入文件或使用 GPU：

- 代码根：`/home/user/etsf_smolvla_piper_schema6_target_development300_prereg_code_r10_20260828`
- 脚本 SHA256：`fc02e59a1fc7eab7c32c673fa8cc4cd476004d57a0f1b4a95f69f436b8bd5ab5`
- 预注册文件：`/home/user/etsf_smolvla_piper_schema6_target_development300_prereg_r10_20260828.json`
- 文件 SHA256：`917a8bbb0a1627881a0ba1dce6857dcefdcf07ba6989383790219fb93eedae52`
- logical SHA256：`bd65a736e1f0ae271937494506b4fd14b30a2de117dc559468b9fe45f511e28f`
- partition SHA256：`e78ef3fd221171e2a2bf1b0fa8a700d1984b2c131514733436da25573e5548ef`

输出为 `0444`，精确冻结 train80 / internal30 / formal190 和每组四候选；其 `collection_authorized=false`，因此它本身不会触发采集。后续仍需 r9b reset 身份、evaluation-disjoint metadata attestation 和新的 runner authority 才能授权服务器执行。
