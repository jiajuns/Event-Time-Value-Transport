# ETSF 多策略 / 多本体验证矩阵

> 审计快照：2026-08-27。4090 主机仅做了文件、manifest、HDF5 schema 和既有结果的
> 只读核查；本次没有启动训练或采集。正在进行的 OpenVLA 分支采集数量会继续变化，正式
> 实验必须以冻结后的 manifest SHA256 为准。

## 1. 现有证据能支持到哪里

| 资源 | 规模与接口 | 协议状态 | 可用于 | 不可用于 |
|---|---|---|---|---|
| OpenVLA factual rollout | Piper / `move_can_pot`，150 episodes、1124 query transitions，hidden 4096、action `25×14`、event/object/proprio | 有效，episode-level split | 事实事件动力学预训练 | 反事实候选排序增益 |
| OpenVLA candidate schema-v1 | train 30 + val 25，baseline/oracle 分别为 `12/20`、`3/5` | **无效**：同 seed 分支语言被重采样，目录有 `INVALID_PROTOCOL.json` | 只能诊断候选多样性 | 任何因果候选比较或论文结果 |
| OpenVLA dense event branches | 4090 上 schema-v4 采集中；快照时 15 groups、60 candidates，baseline/oracle `4/9`，53/60 duration observed；后续可能升级 v5 | 语言固定、post-hidden/event/object 齐全，但尚未完成/冻结 | 反事实 event/time/object/ranking 微调 | 当前就宣称成功率提高 |
| SmolVLA fixed-language candidates | Aloha / `move_can_pot`，`50×14`、expert hidden 720；有效 train 28、val 10、sealed test 10；另有新增 train 44、val 19 | schema-v2 有效；720-D hidden 每候选经过10步 denoise且实测不相同；旧 `official_exec50...` 目录有 INVALID 标记 | SmolVLA 原生 direct-Q / pairwise 候选排序 | 训练 event/time/object dynamics（没有密集标签） |
| SmolVLA shared-prefix schema-v5 collector | Aloha；960-D contextualized VLM prefix state、`50×14` action、逐 step object/proprio 与 continuation query | collector、fail-closed validator、CPU synthetic→trainer 接口已完成；4090 hook/环境 smoke 尚未执行 | 采集 P2 所需 dense supervision | 当前即宣称真实数据可用、成功率改善或跨策略迁移 |

SmolVLA 的 P2 前置契约还包括：HDF 的 event-spec/source SHA、共享状态内容寻址 id、
候选 root state 逐位相同、noise seed 唯一且至少一个候选动作确实改变。trainer 只能从
同一 960-D `state_contract` 的 factual checkpoint 初始化；在线 adapter 也必须回报同一 id。
| Stage3 source/target | 六任务；Aloha+ARX-X5 各 `6×50` source，Piper+UR5-WSG 各 `6×20` target；5 seeds | 开发性 LOEO，Piper/UR5 已被反复查看 | 跨本体事件/时钟机制 | 动作条件 reranking、确认性零样本结论 |
| Franka/Panda public metadata | 10 tasks ×100 trajectory pickle；与 Stage3 重叠四任务：adjust/beat/lift/place | 只有 joint path/status；已参与速度审计，尚未进入 Stage3 训练 | 新 body 的 clock 外部验证候选 | 不重放就做对象事件/动作效果监督 |

已有 SmolVLA sealed test 的事实结论是：冻结 guard 后 baseline 与选择器均为 `1/10`，差值
`0.0`、95% bootstrap CI `[-0.3,0.3]`，candidate AUC `0.355`。这次测试没有证明成功率
提高。初始 validation 的 `1/10→3/10` 不能替代 sealed test。扩展版在 19 个新 validation
groups 上 AUC 提高到约 `0.65–0.67`，但 guard 选择零个非 baseline 候选，也不能声称改善。

Stage3 terminal LOEO 的机制证据是：Piper duration MAE `14.727→4.162`、same-event AUC
`0.6119`（30 对）；UR5-WSG `18.066→8.656`、AUC `0.7917`（26 对），五个 seed 的
duration/event-MC 均改善。但正式状态仍为
`inconclusive_insufficient_matched_event_pairs`，并且 `stop_before_critic_integration=true`。

## 2. 当前最关键的结构缺口

多策略适配必须分成两个独立接口：

```text
native observation/history ── StateAdapter ── shared state contract
native action candidates   ── PolicyAdapter ─ canonical action contract
embodiment/action frame    ── BodyAdapter   ─ shared action effect
embodiment timing          ── ClockAdapter  ─ beta / reference-step scale
current object/perception  ── PredicateAdapter ─ calibrated predicate vector
```

仅有 `PolicyAdapter` 不够。当前 OpenVLA 世界模型硬契约为 4096 维 hidden；SmolVLA 文件是
720 维 action-expert hidden，而且每个噪声候选的 hidden 不同。把它 padding 到 4096、取
candidate-0 或学习一个未经验证的线性投影，都会把策略和动作噪声泄漏进“共享状态”。

可验证的 SmolVLA 路径分两步：

1. 用 schema-v5 collector 采集每个 query 在 flow denoise 前的 960-D contextualized VLM
   prefix state，先训练 `state_input_dim=960` 的 SmolVLA 原生事件模型；
2. 原生预测与 rerank 通过后，再用同图像/语言/任务的配对状态训练
   960→shared-semantic StateAdapter，并在同本体数据上验证跨策略，不能用无监督 padding
   或未验证的线性投影替代。

现有插件已经把 `StateAdapter` 和动作 `PolicyAdapter` 分开；identity state adapter 只验证
维度，不创造跨策略对齐。未校准 state/action/body/clock adapter 均只能 monitor，不能
授权 rerank。

结构化事件还多一个独立的 `PredicateAdapter` 责任：在线输入必须从当前 query 状态按训练
manifest 的 predicate derivation 与 task calibration 得到，不能缺失时补零。现有训练谓词来自
仿真对象位姿；新本体若改变对象坐标系、goal/anchor 定义或运动阈值，必须重新标定并验证。
实机或只有图像的策略还需要单独校准 predicate detector。因此相同的五个谓词名称提供的是
可共享语义接口，不等于已经实验证明跨本体观测迁移。

## 3. 分阶段可执行矩阵

| ID | 冻结核心 / 训练部分 | 训练域 | 适配域 | 密封测试域 | 回答的问题 | 当前状态 |
|---|---|---|---|---|---|---|
| P0 | OpenVLA actor 冻结；训练 event ensemble | OpenVLA-Piper factual + 有效 dense branch train | 同策略 val，仅调 temperature/guard | 同策略全新 seed，≥50 groups | 动作条件事件模型能否改善原策略 | 等 dense schema v4/v5 完成 |
| P1 | SmolVLA actor 冻结；direct-Q | Smol-Aloha fixed-language train | val 10 | 已用 test 10 | 只换候选生成算法是否已有增益 | 已完成，**未改善** |
| P2 | Smol 原生 event core | 新采 Smol 960-D shared-prefix + dense branch | 新 val，只校准 Smol state/action adapter | 全新、未看 seed ≥50 | SmolVLA 上能否准确预测 event/time/object | schema-v5 collector/CPU loader 已完成；待4090 smoke与正式采集 |
| X-P | shared event core 冻结；仅 State/Policy adapter | 一个策略 | 同一 body/task 的另一策略少样本 `N={5,10,20,50}` | 另一策略新 seed | 真正跨策略迁移 | 当前 OpenVLA=Piper、Smol=Aloha，**策略与本体混杂，不能做** |
| E0 | Stage3 shared semantic 冻结；仅 clock beta | Aloha+ARX-X5 六任务 | Piper 或 UR5 每任务前5条 | 每任务后5条 | 事件时钟是否跨本体 | 开发机制通过，确认门未过 |
| E1 | 同 E0，Franka 完全不进 shared training | Aloha+ARX-X5 | Franka 四个重叠任务少样本 clock | 预注册其余 seed | 新 body 是否复现 clock 改善 | 需重放 Panda joint paths以取事件/对象状态 |
| X-E | action-effect core 冻结；仅 Body+Clock adapter | 一个或多个 body 的 dense candidate branch | 新 body `N={5,10,20,50}` | 新 body ≥50 groups | 动作条件世界模型能否跨本体提高成功率 | 四本体 Stage3 没有候选动作，尚不能做 |
| X-PE | shared core 冻结；State/Policy/Body/Clock adapters | 至少 `2 policies × 2 bodies` | 留出一个 policy-body cell | 全新 cell seeds | 同时跨策略和本体 | 最终实验，当前数据不足 |

### 推荐实际顺序

1. 先完成 P0：这是当前数据和工程最接近成功率闭环的一格。
2. 同时修订 Smol collector，保存候选共享 state 与 schema-v4/v5 等价的 dense labels，再做 P2。
3. 选择一个共同本体完成 X-P；在此之前不能用 OpenVLA-Piper 对 Smol-Aloha 的差异代表
   “跨算法”。
4. 用 Franka/Panda 做 E1 外部 body 复验；若只能使用已汇总查看过的数据，报告为 external
   holdout，而不是 untouched sealed confirmation。
5. 只有 X-P 与 E1 分别通过后再做 X-E/X-PE，避免一次实验同时混入 state、policy、body、
   clock 四种域偏移而无法归因。

## 4. 每一格必须冻结的比较与门槛

### 4.1 公平比较

- 相同 simulator requested/resolved seed、instruction、候选数、执行 horizon 和 continuation actor；
- baseline 必须由 `candidate_name=deterministic` 显式确定，不能依赖数组排序；
- 同一候选集合比较 B0 actor、logprob、direct-Q、event-only、event+clock、完整 ensemble；
- adapter 参数量、校准样本量和推理延迟单独报告；shared core 的训练前后 SHA256 必须相同；
- split 以 `(task, body, policy, requested_seed, resolved_seed)` logical key 分组，不能按 candidate
  或 query 随机拆分；
- 旧 INVALID_PROTOCOL 目录在 loader 层硬拒绝，不与 fixed-language 数据拼接。

### 4.2 预测准确性门

在 validation 冻结模型选择，再一次性打开 sealed test：

- next-event：macro-F1、balanced accuracy、confusion matrix；必须超过 current-event/self-loop
  和 event-frequency baseline；
- reach/success：ROC-AUC、PR-AUC、Brier、ECE；低正例率必须报告 PR-AUC；
- duration：observed MAE、删失 NLL、coverage；与 event×body median 比较；
- object delta：raw-meter MAE/NLL，不能只报标准化空间；
- uncertainty：risk-coverage/AURC；高 uncertainty 子集错误率应单调增加；
- ranking：within-group pair accuracy，并按事件阶段分别报告，防止 e0 占比掩盖晚期失败。

### 4.3 成功率改善门

validation 上只有同时满足以下条件才写 `guard.enabled=true`：

- 至少 10 个非 baseline proposal、coverage ≥10%；
- scoring 只从预注册的 7 项 success-only/progress 小网格中选择并完整记录，禁止连续扩展；
- gain 与 uncertainty 阈值只用 validation 的固定至多 3×3 分位数网格选择；
- guarded paired success delta 的 90% 下界不低于 0；
- 相对 baseline 的 harmful rate 不超过预注册的 10%。

sealed test 报告 paired delta、episode bootstrap 95% CI、exact McNemar/sign test、改变次数和
oracle gap。确认性“提高成功率”要求点估计 `>0` 且 95% 下界 `≥0`；样本不足时写
inconclusive，不能用 oracle success 代替模型选择成功率。

### 4.4 迁移门

- `N=0` 仅是诊断；论文主张采用 `N={5,10,20,50}` adaptation curve；
- 共享 core 冻结，只有对应 adapter/clock 可更新，并做逐参数 hash 审计；
- 与 target-only、full-finetune、event-index、body median clock 比较；
- 跨策略必须同 body/task；跨本体必须同 policy/task，分别消除混杂；
- 至少一个未参与架构/阈值开发的新 policy 或新 body 才可称 confirmatory transfer。

## 5. 资源结论

当前已经具备：OpenVLA-Piper 的事实监督、正在补齐的有效反事实监督、SmolVLA-Aloha 的
候选排序监督、SmolVLA 共享-prefix schema-v5 collector/CPU schema 验证，以及四本体六任务的
事件/时钟开发数据。当前仍缺：**4090 实采的** SmolVLA 共享状态与密集事件标签、同本体双
策略数据、以及带动作候选的新本体数据。因此近期可以完成
“OpenVLA 上准确事件预测与 guarded 成功率改善”和“Stage3 clock 外部 body 复验”，但还
不能把二者相乘后直接宣称一个模型已同时跨策略、跨本体提高成功率。
