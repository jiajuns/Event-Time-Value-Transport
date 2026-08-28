# Schema6 Piper dual-provider routerdev320 标签盲预注册 v1

## 1. 结论与适用范围

`scripts/preregister_smolvla_piper_schema6_routerdev320_v1.py` 为当前
`body_agnostic_adapter` 与 `body_conditioned_adapter` 双 provider 新建一份完全独立的
Piper router-development 数据预注册。它使用 Python 标准库，不接受任何输入文件，不读取
reset、trajectory、HDF5、标签、adapter training、Formal190 或 evaluation400 数据，也不执行环境、
策略或模型。

本协议选择 Piper-only 320 组，而不立即建立跨本体 300×2 数据集，是因为当前 candidate
provider 是 `SmolVLAPiperAdapter`：target body 固定为 Piper，body-conditioned 前向固定使用 target
body row。让同一个 candidate provider 在 Aloha 与 Piper 两个 context 上运行，当前没有可验证的
generic adapter/loader 支撑。routerdev320 可以先回答“在 Piper 上，逐头使用完整本体 adapter 是否
优于 body-agnostic 回退”，但不能单独证明多本体普遍改善。

这是一个新协议族的 v1，不是历史 development300 v1 的升级或重新解释。历史的 adaptation110、
Formal190 以及 evaluation400 的成员和权限均保持不变。

## 2. 唯一物理 split

预注册固定：

| 项目 | 数量 |
|---|---:|
| 物理 split | `dual_provider_router_development` |
| Piper execution groups | 320 |
| identity freeze 后的 semantic-reset clusters | 320 |
| 每 cluster execution groups | 1 |
| 每组 candidate accounting | 4 |
| 总 candidate accounting | 1280 |
| adapter train/internal 成员 | 0 |
| Formal190 成员 | 0 |
| evaluation400 成员 | 0 |

全部 320 个 requested seed 来自新 namespace
`schema6_piper_dual_provider_routerdev320_20260829_v1` 和默认独立 range
`2026084500..2026084819`。已冻结的 development300 range `2026082800..2026083099`、Source
dense260 candidate range `2026083500..2026083899` 与早期单组 seed 均被显式排除。seed 只按
namespace SHA-256 排序，成员
关系不依赖 reset 稳定性、候选效果、success、event、duration、recovery、object effect 或任何模型
输出。

本协议要求将来精确处理全部 320 组。不得按 support 或指标提前停止；requested/resolved seed
必须逐项相等；失败或不稳定 seed 不得在本协议内重试、替换或移动；不可行 candidate 可以按冻结
collector 语义记为 censored，但不能换成其他 candidate。增加数据只能新建 create-once 版本。

## 3. Canonical nested OOF 尺寸

身份解析完成后，独立 identity authority 必须在打开任何 collection label 之前，以
`semantic_reset_cluster_id` 为唯一折分单元物化 canonical 计划：

```text
sort(unique semantic_reset_cluster_id)
fold = zero_based_sorted_index mod 5
```

320 可以得到精确的外层尺寸：

- 5 个 outer folds；
- 每折 heldout 64 groups/clusters；
- 每折 training 256 groups/clusters；
- 每个 group/cluster 外层恰好 heldout 一次。

每个 outer-training 域内再次按同一算法做五折：

- inner heldout 为 `[52, 51, 51, 51, 51]`；
- inner training 为 `[204, 205, 205, 205, 205]`；
- 最小 innermost training 域是全量的 `204/320=0.6375`。

320 的规划证据显式冻结但不是数据保证：公开单候选 success Wilson 下界取
`p=0.0981819861`；仅在“四候选 IID”规划假设下，组级至少一次 success 概率为
`q=1-(1-p)^4=0.3385825866`。5 outer × 5 inner 共 25 个 innermost scope，按 union bound 控制
familywise failure 不超过 0.05 时，每个 inner scope 至少需要 203 组：202 组的 bound 为
0.0501255964，203 组为 0.0433199824。按最内层约 64% 全量反推得到分析最小量 318，再工程取整为
320；实际 canonical 320 计划的最小 inner training 是 204。

IID 假设、公开成功率和该样本量都不保证真实标签一定过门。真实 support 必须在采集后重新计算；
caller 自定义折、按标签平衡折、跨 cluster 拆分或用 Formal190/evaluation400 补折均不允许。

## 4. 六头 support 门

每个门必须同时按独立 execution group 与独立 semantic-reset cluster 计数，并在 global、每个
`body×actor` context、每个 outer-training 域以及每个 inner-training 域通过。本 v1 精确只有一个
context：Piper + `smolvla_robotwin_aloha-trained__piper-zero-shot`。identity freeze 还必须绑定
body contract SHA、actor contract ID 与 actor contract SHA。

| head | 类别 | 每类最小独立 group/cluster |
|---|---|---:|
| post-event | `e0/e12/e3/e4/eK` | 10 |
| observed next-event | `e12/e3/e4/eK` | 10 |
| duration | observed / censored | 10 |
| success | positive / negative | 50 |
| conditional recovery | positive / negative | 10 |
| object effect | nonzero / near-zero | 50 |

recovery 只在 `regress & recovery_observed` 上适用；右删失 non-recovery 不能伪装成负例。未观测或
右删失 success 也不能伪装成失败。object effect 只计有效观测监督。任一头不足时只能禁用该头；
六头未全部通过时不得导出完整 route receipt，也不得事后补 seed。

## 5. 两阶段外部不相交证明

预注册只冻结证明要求，本身不自称已经证明不相交。后续 authority 必须由外部 issuer/signature
验证；自报布尔值或自哈希不够。三个 reference role 缺一不可：

1. `provider_training_closure`：shared-core training 与 adapter train/internal checkpoint-selection
   的完整并集；
2. `formal190`：保持不变的 Formal190 identity set；
3. `evaluation400`：保持不变的 evaluation400 identity set。

必须进行两次证明：

- candidate-pool 阶段，在 reset authority 前证明 requested seed 与
  `semantic_reset_request_sha256` 的交集为零；
- selected-resolved 阶段，在 collection authority 前复用同一批 reference commitments，并证明
  requested seed、resolved seed、`semantic_reset_cluster_id` 与 `execution_group_id` 的交集均为零。

外部 reference 的 identity 或 label 原值不得写入公开 preregistration。任何 role 缺失、commitment
变化或交集非零，都必须在 reset/collection 之前失败关闭。

## 6. 2×5 provider prediction 必须先于标签

router target labels 只有在以下顺序完整成立后才可物化：

1. 冻结完整的两个 provider，每个 provider 精确五个成员；
2. 两者共享 core lineage、training execution/semantic identities、member index 与 training seed，
   但 provider artifact SHA 必须不同；
3. 冻结只含因果输入与 sample order 的 label-blind input view；
4. 对全部十个 checkpoint 执行前向；
5. create-once 冻结原始 prediction tensors 与 checkpoint→tensor forward receipt；
6. 才可授权物化 post/next/duration/success/recovery/object target 与 applicable masks；
7. 使用标签打开前已经绑定的 calibrator 实现和 canonical fold plan 运行 nested OOF。

prediction 进程不得接受 target-label、Formal190 或 evaluation400 路径；prediction commitment 后
不得按标签删行、换序或改变成员。rank 不属于双 provider 六头路由，必须继续使用 actor baseline。
routerdev 结果最多只能作为后续 Formal190 authority 的候选依赖，不能直接授权 evaluation、部署或
性能声明。

## 7. 当前 capability

该预注册的 reset、policy query、simulation、真实机器人、collection、training、prediction、label
open、calibration、promotion、deployment 和 performance/transfer claim 权限全部为 `false`。
prediction commitment 之后的 nested-router 拟合仍要求单独 authority；预注册只冻结这个先后关系，
自身不授权拟合。
它也没有读取 input file、trajectory、HDF5、outcome 或 label。

要实际采集，仍需新增并分别审计：

- routerdev320 identity materializer；
- candidate/selected 外部 disjointness attestor bundle；
- routerdev320 sealed collection worker/runner；
- causal input-view materializer；
- 2×5 provider forward runner；
- prediction-gated label materializer；
- 绑定真实 forward receipt 的 route freezer；
- 面向未见 Formal190/evaluation400 batch 的 inference router。

现有 development300 identity materializer/runner 硬编码 300 与 80/30/190，不能直接拿来解释本
协议；应另建新格式，旧 v1 文件保持不变。

## 8. 生成、重建校验与输出权限

```bash
python3 scripts/preregister_smolvla_piper_schema6_routerdev320_v1.py \
  --output /absolute/non_sensitive/routerdev320_preregistration_v1.json
```

可选 `--seed-base` 仍必须落在合法、非重叠的数值 namespace。程序会从 seed base 完整重建整个
文档，逐字段比对确定性合同，再以 canonical JSON SHA-256 签名。输出使用 create-once 硬链接发布，
权限固定为 `0444`；已有文件或 symlink、以及 fresh、confirmation、formal、evaluation、
adapter-train 等敏感输出 namespace 都会失败关闭。

定向测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /tmp/etsf-python310-readonly -m pytest -q \
  tests/test_preregister_smolvla_piper_schema6_routerdev320_v1.py
```

测试只构造内存 JSON 和临时输出路径，不连接远端，不 reset，不运行策略，不读取真实 HDF/标签。
通过这些测试只能证明预注册合同与失败边界可重建，不能证明已经获得监督数据、模型准确率提升、
跨本体改善或任务成功率提升。
