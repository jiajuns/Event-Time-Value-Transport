# RoboTwin2 跨本体共享事件头正式测评协议 v3

> 版本日期：2026-08-31
>
> 适用主线：RoboTwin2.0 `move_can_pot` 五本体 LOBO，冻结 actor，外置 critic 做 best-of-N 候选重排。
>
> 本文定义最终论文口径，不代表五折训练、RAC/WCM 对照或闭环实验已经完成。

## 0. 最终只回答一个主问题

在完整留出一个机器人本体的 critic 监督后，只使用另外四个本体训练共享事件头，能否在留出本体上对同一个冻结 actor 的同一候选池进行重排，并提高完整任务成功率？

主结论必须由以下闭环指标给出：

```text
CrossEmb-DeltaSR@4
= SR(冻结 actor + 留出本体 ETSF critic，从 4 个候选中选择)
  - SR(冻结 actor，执行 candidate-0)
```

AUROC、AP、Brier、事件 F1、时长误差、对象效果误差和 AURC 只解释 critic 为什么有效或为什么无效，不能替代 `DeltaSR` 证明迁移成功。

论文正文只需要两张核心表：

1. **结果表**：`DeltaSR`、`DeltaStage`、候选覆盖/选择效率、延迟和显存；
2. **机制表**：成功校准、事件、时长、对象效果和不确定性指标。

不要为了“指标多”继续增加模型门控。新增指标优先使用训练结束后的冻结预测与已有 outcome 离线计算；除独立闭环、候选真值和延迟基准外，不应重新采集数据。

## 1. 评测范围与主张边界

### 1.1 冻结范围

| 项目 | 正式设置 |
| --- | --- |
| benchmark | RoboTwin2.0 |
| task | `move_can_pot` |
| bodies | `aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5` |
| conditions | `clean`、`randomized` |
| critic split | 五折 leave-one-body-out（LOBO） |
| seeds | 每个 `body × condition` 100 个预注册 seed |
| primary pair | `actor_n1` vs `ETSF_n4` |
| secondary pools | 嵌套的 `N=1/4/8`，且 `N4 == N8[:4]` |
| actor | 同一 checkpoint，全程冻结 |
| target adapter | 仅解析式、无标签的 state/action frame adapter |

完整闭环规模为：

```text
5 bodies × 2 conditions × 100 paired seeds = 1000 paired initial conditions
```

若在每个初态上分别执行 `actor_n1`、`critic_n4` 和 `critic_n8`，则是 3000 条闭环 rollout；另行执行的 query-0 候选分支只用于候选覆盖和 oracle 诊断，不混入闭环 `DeltaSR`。

### 1.2 “跨本体”的准确含义

当前实验可以证明：

> critic/event head 对未见过 critic 标签的机器人本体发生迁移，并可作为冻结策略的外置可插拔重排模块。

如果统一 actor 的训练数据已经包含留出本体，则不能写：

> actor 与 critic 联合零样本迁移到新本体。

报告必须同时给出 actor 训练本体清单和每折 critic 训练本体清单。`actor exposure` 与 `critic exposure` 是两个不同问题。

### 1.3 五折闭包

每一折必须同时满足：

- 留出本体的 branch outcome、事件标签、normalizer、checkpoint selection、calibration、score 权重和阈值均不进入训练或模型选择；
- 训练本体恰好是另外四个本体；
- 留出本体只使用解析式、无标签坐标适配，不学习 body embedding、clock row 或 action stem；
- actor checkpoint、event spec、候选生成、执行步长、N、tie-break 和 seed roster 在正式 outcome 前冻结；
- baseline 与 critic 使用相同 requested seed、等价 reset 和相同的有序候选前缀；
- crash、timeout、不可行动作与 task checker 失败按 intention-to-treat 保留，不能换 seed 或重试到成功。

任意一折不完整时，只能报告“当前完成折”，不能称为五本体 Cross-Emb 结果。

## 2. 数据与监督

### 2.1 官方 expert 数据

RoboTwin2 官方 `clean_50` / `randomized_500` 专家演示可用于 actor、解析式坐标适配和成功轨迹状态分布参考。它们不是完整 critic 监督：没有失败标签的专家 episode 不能被当作失败样本。

### 2.2 共享头的监督单位

正式 critic 监督来自 simulator 中同一 root 的真实候选分支执行。每个 decision group 至少包含：

- `body / condition / requested_seed / episode / query / candidate_index`；
- 当前 canonical state、当前事件 `e_t` 和 actor action chunk；
- post-event、next-event、事件边界及 observed/censored mask；
- native task checker 的终局成功/失败；
- terminal event、阶段进度和目标距离进展；
- moving object 的平移与旋转变化；
- root、候选池、critic checkpoint 和结果文件的不可变哈希绑定。

监督目标对应当前共享头：

```text
(e_t, canonical state, canonical action effect)
    -> p(next event)
    -> p(holding time / censoring / competing event)
    -> p(success / failure / recovery / regression)
    -> object SE(3) effect distribution
    -> aleatoric + ensemble epistemic uncertainty
```

## 3. 必须比较的方法

| ID | 方法 | 是否部署可用 | 回答的问题 |
| --- | --- | --- | --- |
| A0 | Actor candidate-0 (`N=1`) | 是 | 冻结 actor 的真实基线 |
| A1 | Uniform@N | 是 | critic 是否优于在相同候选池随机选择 |
| M | ETSF v13 shared event head | 是 | 结构化跨本体 critic 是否有效 |
| B1 | RAC（relative action critic） | 是 | 相对标量偏好是否已经足够 |
| B2 | WCM matched future-latent critic | 是 | 非结构化未来 latent 是否已经足够 |
| O | Candidate oracle@N | 否 | 候选池可利用上限；只能离线诊断 |

三条训练方法 M/B1/B2 必须使用同一个四本体训练折、同一 branch 数据、同一候选池和可比参数/训练预算。外部论文中的数值不能作为本项目 baseline 数值。

两个后续对照只有在所需信号真实存在时再补：

- **Actor-confidence@N**：若 actor 能输出定义清楚、可复现的每候选 density/score，则报告 actor 自置信度排序。MG-Select 的 masked-reference token KL 依赖自回归 token 分布，不能原样声称已迁移到连续 flow-matching SmolVLA；需要先定义并验证连续动作版本。
- **Frozen-hidden probe**：若采集时保存了 actor pre-action hidden state，可在相同 LOBO split 上训练线性/浅层 success probe。当前只保存 `state27 + action + outcome` 的 branch 文件不能事后凭空恢复 hidden state。

DeepHit/Neural Fine-Gray 是时长头结构对照，不是跨本体成功率指标；V-GPS-style offline Q 可作为更强的标量 Q 对照，但不能替代当前同预算 RAC。

## 4. 第一层：闭环主指标

### 4.1 单元格成功率与配对提升

令 `y^m_{b,c,s} ∈ {0,1}` 表示方法 `m` 在留出本体 `b`、条件 `c`、requested seed `s` 上的完整任务结果：

```text
SR^m_{b,c} = mean_s y^m_{b,c,s}
DeltaSR^m_{b,c} = mean_s (y^m_{b,c,s} - y^actor_{b,c,s})
```

主指标使用十个 `body × condition` 单元格等权宏平均：

```text
CrossEmb-DeltaSR@m = mean_b mean_c DeltaSR^m_{b,c}
```

必须同时报告：

- 十个 cell 的 `SR_actor`、`SR_method`、`DeltaSR` 和样本量；
- 五个 body 的等条件宏平均；
- 全局等 body/condition 的 `CrossEmb-DeltaSR`；
- success-success、success-fail、fail-success、fail-fail 四格配对计数。

主比较固定为 `ETSF_n4 - actor_n1`。`ETSF_n8 - actor_n1` 和 `ETSF_n8 - ETSF_n4` 是次要扩展，不允许看到结果后替换主比较。

### 4.2 置信区间与检验

- 主区间：20,000 次 **requested-seed cluster paired bootstrap**，同一抽样中保留该 seed 下所有 body/condition 配对；
- 稳健性区间：`body × condition` cluster bootstrap；只有 10 个 cluster，必须标为敏感性分析；
- cell 内二值配对：exact two-sided McNemar；
- 跨 cell 合并的普通 McNemar 只作描述，不能忽略 cell 结构冒充全局推断；
- 报告点估计、95% CI、discordant pair 数，不只报 p 值。

论文主张“提高成功率”的最低证据是：1000 对完整、主 `CrossEmb-DeltaSR@4 > 0`，且预注册 requested-seed cluster 95% CI 不跨 0。单个本体无需都达到显著，但必须完整披露负向 cell，不能靠宏平均隐藏伤害。

### 4.3 阶段进度

冻结阶段支持：

```text
0.00 未达到
0.25 reach
0.50 grasp
0.75 transport / lift-to-target
1.00 native task success
```

报告：

- `MeanStage` 与配对 `DeltaStage`；
- 每个阈值 `P(stage >= q)` 的阶段到达率；
- requested-seed cluster 95% CI。

`stage=0.75` 仍是任务失败；`DeltaStage` 是低成功率时的灵敏辅助终点，不能把失败改写成成功。

## 5. 第二层：候选池覆盖、oracle 与选择效率

当前 N≤8 数据是“同一个 decision root 下执行不同候选动作分支”，因此正式名称必须是 **CandidateCoverage@N / CandidateOracle@N**，不能称为 SVA 的完整任务 `pass@k`。

对一个 decision group `g` 的前 N 个嵌套候选：

```text
CandidateCoverage@N
  = mean_g 1[max_i<=N success(g,i) = 1]

CandidateOracleStage@N
  = mean_g stage(g, argmax_i lexicographic(success, stage, goal_progress, -index))

OracleRegret@N
  = mean_g [outcome(oracle candidate) - outcome(selected candidate)]
```

同一条可实现候选必须同时定义 oracle 的 success、stage 和 goal progress；不能分别对三个终点取最大值后拼出一条不存在的“oracle 动作”。

Uniform@N 的期望和区间使用固定候选 outcome 离线计算，随机种子预注册。若需要跨 cell 归一化，报告：

```text
HeadroomCaptured@N
  = (SelectedSuccess@N - UniformSuccess@N)
    / (OracleSuccess@N - UniformSuccess@N)
```

约束：

- 同时报分子、分母、group 数和 bootstrap CI；
- 分母绝对值低于预注册阈值时报告 `N/A: no candidate headroom`；
- 不把结果裁剪到 `[0,1]`，负值表示比随机选择更差，大于 1 表示抽样/估计波动或协议错误需审计；
- branch oracle 是单步/单 query 候选诊断，不与完整闭环 `SR` 混为一个 estimand。

若以后要与 SVA 的 `pass@k` 字面一致，必须从同一初始条件采集 N 次**独立完整策略 rollout**，再用组合估计式报告 `pass@k`；当前 raw8 candidate branch 不能替代这项实验。

## 6. 第三层：共享头机制指标

所有机制指标必须按 held-out body 计算，再做等折宏平均，并显式报告 `n_total / n_positive / n_negative / n_censored / class support`。支持不足时输出 `N/A`，不能用 0 填充。

### 6.1 Success / failure

主机制指标：

- **Average Precision / PR-AUC**：类别不平衡下的判别主指标；
- **Brier Skill Score**：`1 - Brier_model / Brier_prevalence`，prevalence 基线只能由训练折估计；
- **Brier decomposition**：Reliability、Resolution、Uncertainty；
- calibration intercept / slope 与 reliability diagram；
- AUROC 仅在双类且建议 `n_positive >= 10`、`n_negative >= 10` 时报告。

固定 10 等宽 bin ECE 可保留为历史兼容项，但不应作为主校准结论。正式补充使用 adaptive-bin ACE，并附每 bin 样本数。校准方法若需要拟合，只能在四个训练本体的 validation lane 内完成，不能用留出本体标签。

### 6.2 事件与进度

- next-event、terminal-event 的 macro-F1、每类 F1、support 和 confusion matrix；
- multiclass Brier；事件具有自然顺序时另报 ordinal ranked probability score（RPS）；
- event transition accuracy 只作辅助，不能替代 macro-F1；
- VOC 仅作为附录可比指标，并限定在留出的成功/专家轨迹、足够密的时间采样上。

当前 execute50 最多只有约 4 个 query 点且大量轨迹失败，不适合把 VOC 当主指标。VOC 对任何保序时间重参数化不敏感，正好说明为什么还需要显式时长头。

### 6.3 竞争风险时长

主时长指标：

- **any-boundary IPCW integrated Brier score (IBS)**；
- **IPCW C-index 或 cumulative/dynamic AUC**，二者至少一个；
- cause-specific cumulative incidence function（CIF）的 Brier/AUC，仅在对应事件支持足够时；
- observed-only MAE 作为直观辅助；
- observed density NLL 与 censored survival NLL 作为 proper loss 诊断。

时间网格由训练折确定并对五个 held-out body 共用，尾部在训练折支持不足前截断；IPCW 的 censoring distribution 也只能由训练折拟合。

普通单事件 D-calibration 不能直接套到 competing-risks 的逐事件 CIF 上。若报告 D-calibration，只能用于 overall any-boundary survival，或采用明确支持 competing risks 的校准方法并写出定义。其他事件简单视为删失的 cause-specific C-index 可以作为判别指标，但必须注明 estimand。

### 6.4 Recovery / regression

分别报告 recovery、regression 和 joint recovery：

- AP/PR-AUC 与 support；
- Brier/Brier Skill；
- calibration curve；
- 不同当前事件 strata 的 macro 结果。

这些指标解释事件模型是否真的学到“失败后恢复”而不是只复述最终成功标签。

### 6.5 对象状态变化

在物理单位中报告：

- translation MAE/RMSE（m）；
- rotation geodesic MAE/RMSE（degree 或 rad，全文统一）；
- goal-distance progress MAE/RMSE；
- 若输出分布：Gaussian/mixture NLL 和 50%/90% interval coverage；
- 按 moving object 与事件阶段分层的 support。

只报标准化空间误差不能支撑“对象效果建模准确”。

### 6.6 不确定性与选择风险

- selected-failure rate vs coverage；
- oracle-regret vs coverage 及其 AURC；
- success/error Brier risk-coverage；
- harmful-selection rate：baseline 成功但 critic 选择失败的配对比例；
- ensemble epistemic 与 aleatoric 分量分别报告，不只给总方差。

AURC 越低越好。coverage 阈值必须在训练折 validation 上冻结；留出本体只做评估。

## 7. 第四层：推理成本

每个 `N ∈ {1,4,8}` 在同一 4090、同一 batch、同一图像缓存策略下报告：

- actor candidate generation latency：p50 / p95；
- canonical adapter latency：p50 / p95；
- critic scoring latency：p50 / p95；
- end-to-end decision latency：p50 / p95；
- 每 query actor forward 次数、critic forward 次数和 candidates/s；
- peak allocated/reserved GPU memory；
- 相对 `N=1` 的 wall-time 倍率。

预热次数、正式重复次数、同步方式（CUDA synchronize）、精度、batch size 和 GPU 型号必须写进报告。延迟不能从训练日志推测，应做独立冻结 benchmark；它不需要新 rollout。

## 8. 统计与报告规则

1. **一个主比较**：`ETSF_n4 - actor_n1` 的 CrossEmb paired `DeltaSR`。
2. **次要比较**：ETSF N8、RAC、WCM、Uniform、N4/N8 扩展；统一标成 secondary。
3. **不借外部数字**：外部论文只定义比较方式，不能把其成功率与本项目相减。
4. **不删失败**：缺失、crash、timeout 和 infeasible 进入 intention-to-treat；同时单列基础设施失败率。
5. **不挑折**：五折全部披露；macro 等权，micro 只作附录。
6. **不混数据层**：closed-loop episode、query-level branch、frame/event prediction 分表报告。
7. **不靠阶段救主结论**：binary SR 是主终点，stage 是 supporting endpoint。
8. **不把诊断叫迁移**：AUC/Brier/MAE 改善只能写“机制改善”，不能写“成功率提高”。

推荐的论文结论判定：

- 若主 `DeltaSR@4` CI 下界大于 0：可以主张共享 critic 提高留出本体成功率；
- 若点估计为正但 CI 跨 0：只能写趋势，继续增加预注册 paired seeds；
- 若 oracle headroom 近 0：候选生成是瓶颈，不能由 critic 训练解决；
- 若 oracle headroom 高但 selection efficiency 低：共享头/utility 是瓶颈；
- 若离线 AP/Brier 好但闭环无提升：优先检查 distribution shift、执行频率和候选池绑定，不继续堆门控。

## 9. 最终表格模板

### 表 A：跨本体闭环结果

| Method | N | CrossEmb SR | DeltaSR vs N1 [95% CI] | DeltaStage [95% CI] | CandidateCoverage | HeadroomCaptured | Harmful selection | Latency p50/p95 | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor candidate-0 | 1 |  | reference | reference |  |  |  |  |  |
| Uniform | 4 |  |  |  |  | 0 by definition |  |  |  |
| RAC | 4 |  |  |  |  |  |  |  |  |
| WCM | 4 |  |  |  |  |  |  |  |  |
| ETSF v13 | 4 |  |  |  |  |  |  |  |  |
| ETSF v13 | 8 |  |  |  |  |  |  |  |  |
| Candidate oracle | 4/8 | non-deployable | n/a | n/a |  | 1 by definition | n/a | n/a | n/a |

表 A 下方另列五个 body × 两个 condition 的完整 cell 结果。

### 表 B：留出本体机制结果

| Method | Success AP | Brier Skill | Event macro-F1 | Event RPS | Any-boundary IBS | Dynamic AUC/C-index | Object trans./rot. error | Regret AURC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAC |  |  | n/a | n/a | n/a | n/a | n/a |  |
| WCM |  |  |  |  |  |  |  |  |
| ETSF v13 |  |  |  |  |  |  |  |  |

所有数字后附 support；无定义或支持不足写 `N/A (reason)`，不能写 0。

## 10. 与当前代码的对应关系

| 指标/协议 | 当前入口 | 状态 |
| --- | --- | --- |
| 配对 SR、DeltaSR、DeltaStage、cluster CI、cell McNemar | `evaluate_robotwin2_cross_embodiment_paired_success_v1.py` | 已实现 |
| N1/N4/N8 闭环、五折绑定、候选 oracle/regret | `evaluate_robotwin2_five_body_lobo_n1_n4_n8_oracle_v1.py` | 已实现 |
| 最终 nested 报告物化 | `materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py` | 已实现 |
| success Brier/AP/AUROC、事件 F1/Brier、历史 ECE、AURC | `run_robotwin2_five_body_lobo_offline_ablation_v1.py` | 已实现大部，需改正式汇总口径 |
| Uniform@N、HeadroomCaptured@N | evaluator 扩展 | 待补 |
| Brier Skill/decomposition、calibration slope、adaptive ACE | offline evaluator 扩展 | 待补 |
| IPCW-IBS、dynamic AUC/C-index、cause-specific CIF 评估 | duration evaluator 扩展 | 待补 |
| 原始物理单位对象误差 | offline evaluator/adapter 审计 | 待核对并补正式汇总 |
| 延迟、吞吐和显存 | 独立 benchmark | 待补 |
| 字面意义的 full-rollout pass@k | 新 sealed rollout 协议 | 当前数据不支持，非本轮必做 |

新增离线指标应在 v13/RAC/WCM 全部完成后，对冻结 checkpoint 和现有 branch 数据统一重算。不得为计算新指标重新选择 checkpoint。

## 11. 外部口径来源

- [SVA / Look Before You Leap](https://arxiv.org/html/2607.03751v1)：独立 rollout 的 pass@k、候选扩展与延迟报告；用于区分完整 rollout pass@k 和本项目 decision-branch coverage。
- [RoboMonkey](https://arxiv.org/html/2506.17811v2)：VLA 测试时扩展、verifier 与延迟/吞吐分析。
- [MG-Select](https://arxiv.org/html/2510.05681)：策略内部置信度选择；其自回归 token 定义不能未经修改套到 flow-matching actor。
- [V-GPS](https://arxiv.org/html/2410.13816)：冻结策略加外置 value 的 plug-and-play 动作选择。
- [GVL](https://openreview.net/forum?id=friHAl5ofG)：Value-Order Correlation；本项目仅作进度附录指标。
- [scikit-survival evaluation guide](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)：删失数据的 C-index、cumulative/dynamic AUC 与 IPCW Brier/IBS。
- [Competing-risk calibration analysis](https://arxiv.org/abs/2602.00194)：普通生存校准方法不能无条件直接解释为 competing-risk CIF 校准。

旧协议和外部补充已从当前工作树删除，只保留在 Git 历史中；正式定义以本文 v3 为准。
