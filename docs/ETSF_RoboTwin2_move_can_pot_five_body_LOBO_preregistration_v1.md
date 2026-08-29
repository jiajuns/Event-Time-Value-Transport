# RoboTwin2 `move_can_pot` 五本体 LOBO 主指标预注册 v1

## 1. 目的与边界

`scripts/preregister_robotwin2_move_can_pot_five_body_lobo_v1.py` 在任何数据下载、
ZIP/Pickle/NumPy payload 打开、训练、仿真 reset、策略查询或结果查看之前，冻结一项五本体
leave-one-body-out（LOBO）研究。它只生成一份 create-once、只读 JSON，不执行也不授权真实
下载、训练、评估、promotion、deployment 或“跨本体有效”的经验声明。

主问题是：在相同 held-out body、相同 clean/randomized condition、相同 reset seed、相同 actor
checkpoint 和相同有序四候选集合下，`ETSF best-of-4` 是否比 `actor_baseline` 提高完整任务成功率。
阶段进度和 critic 质量只能作为支持性/诊断性指标，不能替代失败的完整任务成功率主指标。

## 2. 官方来源与固定 revision

官方来源固定为 Hugging Face dataset
[`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，
revision 固定为：

```text
a967b852afa21a9cbf19a198f7e653109042e87c
```

元数据于 2026-08-30 通过该 immutable revision 的
[官方 tree API](https://huggingface.co/api/datasets/TianxingChen/RoboTwin2.0/tree/a967b852afa21a9cbf19a198f7e653109042e87c/dataset/move_can_pot?recursive=true&expand=true)
核对。官方任务页确认 `move_can_pot` 覆盖 Aloha-AgileX、ARX-X5、Franka-Panda、Piper、
UR5-Wsg 五种 embodiment：[RoboTwin 2.0 Move Can Pot](https://robotwin-platform.github.io/doc/tasks/move_can_pot.html)。

`lfs_sha256` 是官方 API 返回的 LFS object OID，即 archive payload 的 SHA-256；同时冻结 Git
blob OID、Xet hash 和精确字节数。脚本运行时不联网，因此未来 `main` 改动不会改变本协议。

## 3. 完整官方 slice

精确清单是五本体各一份 `clean_50` 与一份 `randomized_500`，另加 `demo_clean.zip`，共
11 个文件。总大小为 `21,238,835,871` bytes，即 `21.238835871` decimal GB（约
`19.7802` GiB）。

| 文件 | bytes | LFS SHA-256 |
|---|---:|---|
| `aloha-agilex_clean_50.zip` | 312,581,958 | `fbf5231c5be71405364b09ed718cfca1e07a6509f6d1a801f5922516da0ade09` |
| `aloha-agilex_randomized_500.zip` | 5,561,470,057 | `761ab72a186941be79082f04c05298753b3ffe3c9fd92bc0c3169d84293d75f6` |
| `arx-x5_clean_50.zip` | 218,811,989 | `8af739367ce74d9b982fc37cd712d6e7674de8d26571df40db3bd7d23f50acae` |
| `arx-x5_randomized_500.zip` | 3,653,570,164 | `716429c14998d86865745333f405f1ddff950400fa3c5734cb1120a803320b59` |
| `demo_clean.zip` | 306,859,681 | `dddb150282a009fcb0f2c2e97276269747a00358c6800c67d991f8d4c5d2c0e7` |
| `franka_clean_50.zip` | 183,374,364 | `2ab00e0b65e5bd9c2fdfd5d5355c3990237070f7b1c58c12f0702cfbc0dd4082` |
| `franka_randomized_500.zip` | 3,328,177,600 | `2bd0dc2d4893326d6d1095565f3370e371160cf503988ef333e4233a43677a82` |
| `piper_clean_50.zip` | 212,427,528 | `e902b72d8111065080b0432cf2750b9754e8c84f16909376ecb7c85c0fcd0d2f` |
| `piper_randomized_500.zip` | 3,896,574,298 | `3a3b5ae9dc748c2cda1fc143f1eff5141a1b57defdc052790c6480762f91599d` |
| `ur5_clean_50.zip` | 183,214,565 | `400f1c826d264c1d2e04dab34df2fa7681a5e640b37a21209ac8dd0ec6bc36e4` |
| `ur5_randomized_500.zip` | 3,381,773,667 | `7f60dea19c46b94e3ca50ce9a349e33f1a93d0ad8bbba887fd046669178619dd` |

文件名中的 50/500 是官方文件声明，不等于本预注册已经打开 ZIP 并审计了 archive member 数。
本预注册把 `archive_content_count_verified_without_download=false` 固定写入每一条记录。

`demo_clean.zip` 的内部 body membership 未在数据盲阶段打开核对，因此它只能是 inventory
reference，不能进入任何 LOBO fold 的训练、校准、模型选择或评估；不能依靠猜测将其分给某一
本体。

## 4. 五折 LOBO

fold 顺序和 held-out body 固定为：

| fold | held-out | 训练本体 |
|---:|---|---|
| 0 | `aloha-agilex` | `arx-x5, franka, piper, ur5` |
| 1 | `arx-x5` | `aloha-agilex, franka, piper, ur5` |
| 2 | `franka` | `aloha-agilex, arx-x5, piper, ur5` |
| 3 | `piper` | `aloha-agilex, arx-x5, franka, ur5` |
| 4 | `ur5` | `aloha-agilex, arx-x5, franka, piper` |

每折 prospective training membership 恰好是另外四个 body 的 8 个 archive；held-out body 的
2 个 archive、`demo_clean.zip`、held-out body adapter、held-out outcome 或基于 held-out 的
threshold/checkpoint selection 一律不能进入训练 closure。held-out 官方专家 archive 也不能替代
配对 simulator evaluation。

同一 archive 在某折可以是 training side、在另一折可以是 held-out side，但不同 fold 的 checkpoint
或 metric 不得互相复用。真正训练前仍需另外冻结每折的下载物化收据、payload schema 审计、训练
identity、代码、checkpoint 和选择规则。

## 5. 公共专家数据不是失败监督

这批公开 archive 是 expert demonstrations。仅凭官方文件名、文件数量、任务页所列 data-generation
success rate，不能得到逐 episode 的失败标签，也不能证明 negative class：

- 未到达某个保存点、缺失 tail、未观测 outcome 不能自动标成失败；
- 不能把所有未标注 transition 设为 `success=0`；
- 不能用该 slice 单独训练或验证 success/failure critic；
- positive-only 或 outcome-unknown expert demonstrations 不能证明失败判别、校准或任务成功率提升；
- data-generation success rate 是生成流程的聚合统计，不是本地 archive 的逐 episode 标签。

只有在单独 authority 下载并审计实际 payload 后，训练 body 的 archive 才可能用于 imitation 或有
observed mask 的 causal event representation。任何 success/recovery 监督都必须来自另行冻结且含真实
正负 outcome 的数据合同。

## 6. 配对评估设计

五个 held-out body 全部使用同一 100 个 requested seed：

```text
2026090000 ... 2026090099
```

condition 顺序固定为 `clean` 后 `randomized`；每个 condition 内 seed 升序。每个
`(heldout_body, condition, seed)` 是一个配对 key，两种方法必须从独立但 bit-exact 的相同 resolved
reset 开始。100 个 seed 同时用于所有 body 和两个 condition，不能因 crash、结果或支持不足换 seed、
补 seed、重试到成功或提前停止。

方法配对为：

- `actor_baseline`：使用 ETSF 打分前 actor 有序候选集合中的 `candidate_index_0`；
- `etsf_best_of_4`：在完全相同、预先冻结的四候选集合上选择 frozen ETSF score 最大者；
- ETSF 分数相同按最低 candidate index 打破平局；
- 两种方法必须绑定同一 actor checkpoint、同一 observation/instruction contract、同一 candidate-set
  SHA；
- 不得增加 simulator lookahead、环境 query、隐藏 rollout、方法专属 retry 或 seed replacement；
- 第一项 outcome 打开后，actor/ETSF checkpoint、候选数、tie-break、threshold 和 metric 全部不可改。

为平衡执行次序，偶数 seed ordinal 使用 `actor → ETSF`，奇数 seed ordinal 使用
`ETSF → actor`；两次执行仍各自进行同一 resolved identity 的干净 reset。

总计：`5 bodies × 2 conditions × 100 seeds = 1000` 个 paired trials，`2000` 条 rollout。
本预注册不授权执行它们。执行前还必须固定 simulator/body/actor/ETSF/event-spec/runtime 的文件 SHA
和 exact seed-resolution receipt。

## 7. 指标与统计

### 7.1 唯一主指标

唯一 primary endpoint 是官方 simulator task checker 给出的 full-task success boolean。按以下层级
预注册报告：

1. 每个 held-out body × condition；
2. 每个 held-out body 的 equal-condition macro；
3. 每个 condition 的 equal-body macro；
4. global equal-body-condition macro。

每层至少报告：

- actor baseline SR 和 ETSF best-of-4 SR；
- paired `ΔSR = mean(success_etsf - success_actor)`；
- SR 的 Wilson 95% CI；
- paired ΔSR 的 95% CI；
- actor-only success、ETSF-only success 两个 discordant count；
- exact two-sided McNemar p-value。

ΔSR 的 CI 使用固定 `20,000` 次 paired seed-cluster percentile bootstrap，bootstrap seed 为
`2026090200`。global resampling 时，同一 requested seed 对应的五 body × 两 condition 十个 paired
trials 一起抽样，保持 repeated-seed 结构；不把 1000 行错误地当成完全独立 IID。

McNemar 使用 discordant pairs 上的 exact two-sided binomial test；零 discordant 时 `p=1.0`，不使用
continuity-corrected asymptotic test。

prospective improvement gate 要求 global macro ΔSR 的 95% CI 下界严格大于零、global exact
McNemar `p<0.05`，并且每个 held-out-body macro 和每个 condition macro 的 ΔSR point estimate 均
非负。即使通过，这份 preregistration 本身仍不授权性能声明或部署，必须由独立 verifier 绑定真实
rollout 和 metric receipt。

### 7.2 阶段进度

阶段顺序固定为 `e0 → e12 → e3 → e4 → eK`，terminal max-event progress 映射为
`0, 0.25, 0.5, 0.75, 1.0`。报告两方法 mean progress、paired progress delta/CI 和逐阶段 reach
rate。执行前必须绑定任务专用 canonical event-spec 文件 SHA。

阶段进度是 supporting endpoint：高阶段进度不能把 full-task failure 改写成 success，也不能挽救
失败的主指标 gate。

### 7.3 Critic 诊断

critic diagnostics 只能在全部 1000 paired outcomes 冻结后计算，包括 success Brier/NLL/ECE、
有正负类别时的 AUROC、uncertainty AURC、post/next event accuracy、observed duration MAE 和
observed object-effect error。单类别 AUROC 应为 `null`，不能写成 0 或 1。

这些诊断不得选择 checkpoint、threshold、候选数、route 或 body adapter，也不得替代/挽救完整
任务成功率。所有 missingness 与 applicability masks 必须同时报告。

## 8. Create-once 与 capability

脚本只接受 `--output`，没有 repo、manifest、dataset、label、checkpoint、seed 或 metric 输入参数。
它从源码中的 reviewed constants 完整重建合同并验证 canonical SHA，然后用临时文件、`fsync` 和
hard-link create-once 发布，输出 mode 固定为 `0444`。已有路径、symlink output 或 symlink parent
均失败关闭。

```bash
python3 scripts/preregister_robotwin2_move_can_pot_five_body_lobo_v1.py \
  --output /absolute/non_sensitive/robotwin2_move_can_pot_lobo_v1.json
```

输出 capability 明确为：未联网、未下载、未打开 archive/pickle、未训练、未 reset、未 policy
query、未评估、未读 outcome、未计算指标、未授权 promotion/deployment/cross-body claim。

定向合成测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_preregister_robotwin2_move_can_pot_five_body_lobo_v1.py
```

测试只检查内存 JSON 和临时 create-once 路径，不访问 Hugging Face 或真实数据。测试通过仅证明
协议、官方元数据常量、LOBO membership、配对 seed/order 和失败边界可重建，不能证明模型已训练、
事件预测准确、跨本体可迁移或任务成功率已经改善。
