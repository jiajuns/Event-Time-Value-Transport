# SmolVLA–Piper schema6 标签盲支持与扩样本规划

## 审计边界

本审计只使用以下公开汇总量：Source63 共 63 个逻辑组、每组 4 个候选分支、34 个成功分支、
218 个失败分支。没有接收数据路径，没有打开 HDF5，也没有读取目标、validation 或 evaluation
标签。Source63 是 Aloha/SmolVLA source 数据，其成功率只能作为样本量规划情景，不能证明 Piper
具有相同分布。

纯 CPU 入口为 `scripts/plan_smolvla_piper_schema6_support.py`。默认审计拟议的
现有 v2 的 `adaptation bucket80/target-validation50` 分配；adaptation80 内部以标签盲规则再拆成
train60/internal-validation20，并读取当前代码中的相应门槛含义：

- trainer 在 adaptation 和 internal validation 各要求至少 5 个成功组、5 个失败组和 5 个候选
  outcome-discordant 组；每个事件、observed/censored duration 等还有独立支持门；
- conditional recovery 只有 train、internal validation 各自 recovery 正负独立组均不少于 10 时
  才启用；不足时必须关闭，不进入主 utility/uncertainty；
- calibrator 的 success head 要求 target-validation 中成功组、失败组各不少于 50；event 和
  duration 每侧/每类要求 10 个独立组；abstention 至少保留 50 个 validation 组；
- versioned target calibrator v2 已接入 recovery 标签、五成员 logits 与条件温度，但只有五个
  recovery head 均已训练且 observed regress 中正负独立组各不少于 10 时才允许激活。

## 可由 34/218 确定的量

分支成功率点估计为

```text
p = 34 / 252 = 0.1349206349
```

若仅用于规划、额外假设同组 4 个候选结果 IID，则

```text
P(全失败组) = (1-p)^4              = 0.5600461439
P(全成功组) = p^4                  = 0.0003313702
P(discordant组) = 1-(1-p)^4-p^4   = 0.4396224859
P(至少一个成功)                    = 0.4399538561
```

因此 trainer 的三个 outcome 门同时通过的规划概率为：

| split | 组数 | IID 点估计 | 以分支成功率 Wilson 95% 下界规划 |
|---|---:|---:|---:|
| adaptation train | 60 | 接近 1 | 接近 1 |
| internal validation | 20 | `0.9765095238` | `0.8594555591` |

这些只是 outcome/discordance 支持概率，不是模型性能或任务成功率功效。

不知道 34 个成功分支如何分布到 63 个组时，可以严格推出的组级范围只有：

| 支持 | 数学最小 | 数学最大 |
|---|---:|---:|
| 至少一个成功的组 | 9 | 34 |
| 至少一个失败的组 | 55 | 63 |
| 同时含成功/失败的组 | 1 | 34 |

这说明仅凭分支总数不能声称真实组级支持门已经通过。

## target-validation50 的确定冲突

success calibration 要求正组与负组各 50，而可用组恰好为 50。故每个组都必须既含成功候选又含
失败候选，即 50 个组必须全部 discordant。按上述 IID 点估计：

```text
P(通过) = 0.4396224859^50 = 1.4255908025e-18
```

这是当前 `50/50` 门与固定 validation50 的结构性冲突：数学上不是绝对不可能，但在 Source63
规划情景下等价于不可用。用相同点估计，要以至少 95% 概率取得 success 正负各 50 个独立组，
target-validation 至少需要 134 组。将 34/252 的 Wilson 95% 分支成功率下界
`0.0981819861` 代入，则至少需要 177 组。

## 无法由成功汇总识别的支持

- Event：成功分支数至多给出终态事件支持的必要条件上界；e0/e12/e3/e4 的频率完全未知。因此
  “所有事件门通过”的概率区间仍是从 0 开始，不能以 success 数代替事件覆盖。
- Duration：34/218 不包含 observed/censored duration 数量，概率只能记为未知 `[0,1]`。
- Recovery：没有 regress、persistent recovery 或 right-censor 汇总，真实通过概率未知。即使采用
  最有利的信息情景——每组只有一个互斥、完全平衡的 recovery 正/负标签——internal20 正负各
  10 的通过概率也只有 `C(20,10)/2^20 = 0.1761970520`。达到 95% 需 30 组；真实 recovery 稀疏时
  需要更多，而且必须用专门的 regress/recovery-enriched 采集。
- recovery v2 的实现能力不等于支持已经成立；34/218 仍无法识别 regress/recovery 频率，不能用
  扩大普通成功/失败样本自动补齐。right-censored non-recovery 也不能作为负例。

## 结论与最小规模

当前只有 1 个 target group：

- 现有 v2 的物理分组最低是 adaptation bucket80（内部 train60/internal20）加 formal
  target-validation50，共 130 组，即还缺 129 组；必须预注册扩采，禁止训练。
- IID 点估计下，将 internal validation 扩至 30、target-validation 扩至 134，最低共
  `train80+internal30+target134=244` 组，即 adaptation bucket110 加 target134，还缺 243 组。
- 使用 Source63 分支成功率 Wilson 95% 下界作保守规划，最低为
  `train80+internal30+target177=287` 组，即 adaptation bucket110 加 target177，还缺 286
  组；工程预注册建议向上留余量到 **300 个独立 target
  development groups**。

300 仍不是 event/duration/recovery 通过保证。预注册必须同时指定事件分层、observed/censored
duration 分层和 regress/recovery 富集配额；冻结 split 后逐 split 执行真实支持审计，不通过则
fail-closed。不能查看标签后移动组、改 split seed 或追加到既有 split。

## 使用

```bash
python3 scripts/plan_smolvla_piper_schema6_support.py

python3 scripts/plan_smolvla_piper_schema6_support.py \
  --currently-available-target-groups 1 \
  --output /absolute/non_sensitive_path/schema6_support_plan.json
```

输出为 create-once、`0444` 的签名 JSON，明确记录 `hdf5_opened=0`、各未知概率边界、bare 与
保守样本量以及 `training_authorized=false`。本规划器不修改或替代已冻结的 v2 manifest/split
协议。
