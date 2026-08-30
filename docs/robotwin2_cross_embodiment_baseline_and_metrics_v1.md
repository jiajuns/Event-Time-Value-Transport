# RoboTwin2 跨本体基线与正式指标协议 v1

> 文档日期：2026-08-31。本文以当前仓库中的正式五本体协议为事实来源，并以论文、项目页和
> 官方代码仓库为外部来源。它定义的是实验口径，不代表五折训练或正式闭环评估已经完成，
> 也不授权“已经跨本体提高成功率”的结论。

## 1. 先给出结论

当前能够严格回答的问题只有一个：

> 在 RoboTwin2 的同一个 `move_can_pot` 任务中，将一个机器人本体完整留出后，只用另外四个
> 本体训练共享事件 critic，能否在留出本体上通过冻结 actor 的 best-of-N 候选重排，提高完整
> 任务成功率？

这叫**单任务跨本体迁移**，不是跨任务迁移，也不是通用事件理解。当前任务、五个本体和两种
条件分别是：

- task：`move_can_pot`；
- bodies：`aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5`；
- conditions：`clean`、`randomized`；
- 每个 `body × condition` 使用 100 个预注册 seed；
- 每个 seed 配对执行 `actor candidate-0` 与 `ETSF best-of-4`。

这里留出的是 **critic/event head 的本体监督**，不是 actor 的本体数据。当前统一 EE16 actor
本身可以已经在五本体数据上训练并始终冻结。因此该实验能证明的是“外置共享 critic 在未见 critic
本体上迁移并改善一个已有 actor”，不能证明 actor 对新机器人零样本迁移。若要主张 actor 也能
零样本跨本体，需要像 LAP 那样另做 actor 训练本体完全留出的实验。

因此正式规模为：

```text
5 held-out bodies × 2 conditions × 100 paired seeds
= 1000 pairs
= 2000 closed-loop rollouts
```

当前的论文主指标应是留出本体上的配对 `ΔSR` 和五本体等权的 macro Cross-Emb `ΔSR`。AUROC、
Brier、事件 F1、时长 MAE、对象效果误差和 AURC 都是 critic 诊断指标，不能替代成功率。

## 2. 必须纠正：OXE 没有“每个技能统一 100 次”的协议

[Open X-Embodiment（OXE）][oxe-project]首先是把异构机器人数据统一为 RLDS 的数据集合与模型
资源，不是一个规定统一 seed、统一任务 checker 和统一试验次数的闭环 benchmark。官方
[RT-X 论文][oxe-paper]报告的是在 6 种机器人上合计 3600 次评估，但各实验域来自不同实验室、
机器人和任务协议。官方 [OXE 代码仓库][oxe-code]也没有规定“每个技能必须 100 次”。

所以以下说法必须删除：

> OXE 的标准是每个技能 100 次试验。

正确说法是：

> OXE/RT-X 证明了跨机器人联合训练通常以真实任务成功率评估，但没有给出统一的每技能 100 次
> 协议。本项目选择 100 seeds/cell，是自己的预注册统计设计，不是从 OXE 继承的标准。

不同论文的次数也并不相同。例如：[LAP][lap-paper]对每个任务做 20 次独立真实机器人试验并报
二值成功和有限样本有效的 95% 区间；[MOTIF][motif-paper]在仿真中对每个本体—任务 pair 做
50 次 rollout，在真实机器人表格中每任务做 20 次 rollout。不能把其中任一个数字写成整个领域
统一标准。

## 3. 当前数据和监督到底是什么

### 3.1 官方数据的作用

当前 `move_can_pot` 的官方数据来自 [RoboTwin2 官方数据集][robotwin-data]。官方
[任务页][move-can-pot]确认该任务支持上述五个本体。公开 `clean_50` 和 `randomized_500`
archive 是专家演示，它们适合：

- 训练或校验冻结 actor；
- 建立无标签、解析式的本体坐标适配；
- 提供成功轨迹上的状态、动作和对象运动分布参考。

但专家 archive 不是完整的 critic 监督：未标失败的 episode 不能自动当失败，生成成功率也不是
逐 episode 标签。尤其不能只用正例证明 success/failure、recovery 或候选排序能力。

### 3.2 共享头需要的真实监督

正式共享头监督来自 simulator 中同初态的候选动作分支，而不是把官方专家轨迹中未出现的行为
臆造为失败。一个完整 decision group 至少要包含：

- 当前 canonical state 与当前事件 `e_t`；
- 同一 actor 产生的有序候选动作 chunk；
- 执行动作后的 post/next event；
- 事件持续时间及 observed/censored mask；
- native task checker 给出的成功/失败；
- moving object 的 SE(3) 状态变化；
- terminal 最大事件、阶段进度与目标距离进展；
- 同一 root、seed、body、condition、query 和 candidate index 的完整身份。

这才是有监督数据。官方 expert 数据提供 actor 训练材料，actor 分支执行提供 critic 所需的正负
outcome、时间、事件和对象效果标签。二者不能混为同一种监督。

## 4. 五折 LOBO：怎样才算真正跨本体

对每个 held-out body 独立训练一折：

| fold | held-out body | critic 训练本体 |
| ---: | --- | --- |
| 0 | `aloha-agilex` | `arx-x5, franka, piper, ur5` |
| 1 | `arx-x5` | `aloha-agilex, franka, piper, ur5` |
| 2 | `franka` | `aloha-agilex, arx-x5, piper, ur5` |
| 3 | `piper` | `aloha-agilex, arx-x5, franka, ur5` |
| 4 | `ur5` | `aloha-agilex, arx-x5, franka, piper` |

每折必须满足以下闭包：

1. held-out body 的 branch outcome、event label、normalizer、checkpoint selection、temperature、
   score weight 和 guard threshold 都不得进入训练或选择。
2. held-out body 只允许使用解析式、无标签的 state/action frame adapter；不能为它学习 body
   embedding、clock row 或 action stem。
3. actor checkpoint、候选生成随机性、候选数、动作执行步长、event spec、tie-break 和 runtime
   在第一条正式 outcome 之前冻结。
4. 同一 `body × condition × seed` 的 baseline 与 ETSF 必须从独立但等价的 reset 开始，并使用
   相同的第一次候选集合。
5. crash、infeasible action 和 task failure 都不能通过换 seed 或“重试到成功”删除。

只有五折全部完成，才能把五个 held-out 结果合成 Cross-Emb 指标。只展示最好的一个机器人不是
跨本体结果。

## 5. 七组必须有的有意义基线

下面恰好列出七组主基线。它们分别排除“actor 自己就能做到”“候选池没有上限”“只需一个成功
logit”“只需标量进度”“只需预测未来 latent”“事件结构没有价值”和“根本没有发生迁移”七种
替代解释。

<!-- markdownlint-disable MD013 -->

| ID | 基线 | 公平输入与训练 | 回答的问题 | 当前状态 |
| --- | --- | --- | --- | --- |
| B1 | Actor candidate-0 | 同 actor、同 reset、同候选生成器，固定执行 index 0 | 插件是否真正改善冻结 actor | 正式 runner 已支持 |
| B2 | Candidate oracle | 同一 root 上查看全部候选真实 outcome 后选最好 | 候选池理论上有多少 headroom | 只允许离线诊断，不可部署 |
| B3 | Action-conditioned success-Q | `state + action → p(success)`，同数据同容量 | 结构化事件/时间/对象监督是否优于单一终局值 | 完整离线消融已定义 |
| B4 | Direct scalar progress | `state + action → scalar progress/value` | ETSF 是否只是 VLAC-style 标量进度 critic | 现有轻量入口需接 RoboTwin canonical schema |
| B5 | Future-latent/video → progress | 先预测候选效果 latent，再做 pairwise 或 scalar progress | 是否只需 ProgressVLA/RoboCritic-style 预测后打分 | 需使用同候选、同数据重建，不可借论文数值代替 |
| B6 | Flat next-event model | 直接预测 absolute next event；另做 `−time/−object/−uncertainty` | relative transition、holding time 和 object effect 是否真的有增益 | 需加入同预算 RoboTwin 消融 |
| B7 | Matched target-only from scratch | 只用相同数量的目标本体 groups，参数量和选择预算匹配 | frozen shared core 是否真的带来样本高效迁移 | 跨本体主张前必须补齐 |

<!-- markdownlint-enable MD013 -->

另外两个有用但不计入上述七组的控制是：同候选池的 `Uniform@N` 随机选择 sanity check，以及
允许目标数据更新全部参数的 full-finetune upper bound。它们应报告，但不能替代 B7，因为
full-finetune 的参数量和目标数据利用方式不同。

当前严格 LOBO 是 `K=0` critic target-label 设置，此时 B7 没有目标数据，不能训练，也不能用随机
初始化结果凑数。B7 用于额外的 sealed few-shot 曲线：固定
`K ∈ {1, 3, 5, 10, 50}` 个目标本体 development groups，对比 frozen shared core + adapter
与同 K、同容量的 target-only scratch；正式 confirmation seeds 必须与这些 groups 隔离。
`K=0` 仍由严格 LOBO 结果回答，`K>0` 才回答样本效率。

所有可部署基线必须使用相同候选集合、相同动作执行步长和相同仿真预算。B2 是唯一允许读取候选
真实 outcome 的方法，必须醒目标成“非部署 oracle”。

## 6. 文献方法：哪些能迁移，哪些不能直接搬权重

### 6.1 RoboCritic

这里的 RoboCritic 指 [Robot Critics that Sweat the Small Stuff 项目][robocritic-project]。
该工作用 action-conditioned video model 预测多个候选动作的视觉后果，再由 pairwise progress
critic 选候选；项目页报告了真实任务和 RoboCasa365 上的闭环成功率改善。

可以迁移的部分：

- 冻结 base policy，比较 base 与 critic-guided policy；
- 在同一个 policy 候选池内重排，而不是让 critic 额外获得更好候选；
- 把“预测候选效果再比较进度”作为 B5；
- 同时报 critic 细粒度判别准确率与闭环任务成功率。

不能直接迁移的部分：

- 其输入是图像/预测视频和成对进度判断，ETSF 输入是 canonical state、native action chunk、
  当前事件和时钟；
- 其权重没有 next-event、删失 holding-time、对象 SE(3) 效果或当前共享头的结构化输出；
- 其动作条件视频生成器不是当前 SmolVLA/OpenVLA actor 的原生模块；
- 论文结果不是 RoboTwin 五本体 LOBO，不能横向相减为 ETSF 的跨本体提升。

因此最公平的做法是复用其“candidate future + pairwise progress”方法思想，在相同 RoboTwin
branch 数据和相同 N 上训练 B5，而不是声称直接加载 RoboCritic 权重完成复现。

### 6.2 MOTIF

[MOTIF 论文][motif-paper]把本体无关 action motif 与本体特定执行解耦，并采用
`K ∈ {1, 3, 5, 10, 50}` 的 few-shot transfer。其 `Cross-Emb. Transfer` 是各机器人的
Transfer 指标宏平均；[官方代码][motif-code]已公开。

可以迁移的部分：

- 机器人等权的 macro Cross-Emb 报告方式；
- target pair 与 source pair 分开计算 Transfer/Global；
- `K`-shot 曲线和 matched scratch baseline；
- 显式区分共享时空结构与本体特定执行。

不能直接迁移的部分：

- MOTIF 是 flow-matching policy 的 action-motif 条件生成，不是外置 critic；
- 它需要自己的 VQ motif learner、motif predictor 和 policy decoder；
- 它的 interleaved task mask 不等价于当前“整本体留出”的严格 LOBO；
- 直接接入会同时改变 actor 和 critic，无法归因 ETSF 插件的 `ΔSR`。

MOTIF 最适合作为未来 few-shot policy-transfer 对照和宏平均定义来源，而不是当前共享事件头的可
直接加载 baseline。

### 6.3 LAP

[LAP 论文][lap-paper]把连续末端动作解析成 language-action 进行 VLA 预训练；其
[官方代码][lap-code]与 checkpoint 已公开。LAP 的零样本真实机器人协议是每任务 20 次独立试验、
随机物体位置、二值成功和有限样本有效 95% 区间；fine-tuning 任务还报告分阶段进度。

可以迁移的部分：

- held-out embodiment 的完整任务成功率；
- 随机初态、独立试验、有限样本区间；
- success 与 staged progress 双报但不混写；
- 零样本和少样本适配分开报告。

不能直接迁移的部分：

- LAP 是大规模 VLA 预训练方法，而 ETSF 当前冻结 actor、只训练外置事件 critic；
- LAP action expert、相机输入、语言动作和动作归一化不等于 RoboTwin 双臂 EE16 合同；
- 加载 LAP 后会更换 actor，不能回答“同一个 actor 加 ETSF 是否改善”；
- LAP 的 20 trials 是其论文协议，不是本项目已预注册的 100 seeds/cell。

LAP 应作为跨本体 policy 级外部参考；若将来适配 LAP actor，仍要在同一个 LAP actor 上做
`LAP` 与 `LAP + ETSF` 的配对比较。

### 6.4 MimicGen

[MimicGen][mimicgen-project]把少量 source demonstrations 分成 object-centric segments，经过
空间变换和拼接，在新场景、对象和机器人上生成数据；[论文][mimicgen-paper]与
[代码][mimicgen-code]均已公开。

可以迁移的部分：

- object-centric 分段和子任务边界；
- 把 source 轨迹重定向到不同 reset、对象和机器人以补充成功数据；
- 用统一 subtask/event annotation 连接数据生成与事件监督；
- 生成过程成功率、失败原因和最终保留数据三者分别审计。

不能直接迁移的部分：

- 官方实现围绕 robosuite/robomimic 任务、对象和控制器，不能直接运行 RoboTwin task class；
- 轨迹片段的空间变换不自动解决双臂时序、碰撞、gripper、不同本体 reachability 和 EE frame；
- MimicGen 主要生成成功 demonstrations，不提供可直接用于 success critic 的平衡失败监督；
- 它是数据生成系统，不是 task-success 评估器或 action critic baseline。

所以可行路径是迁移数据生成思想并新写 RoboTwin adapter，再让所有生成轨迹经过 native checker；
不能把 MimicGen 数据量当作 `ΔSR`，也不能把未成功生成的轨迹自动标失败。

### 6.5 VLA-ATTC / Relative Action Critic

[VLA-ATTC][vla-attc-paper]在测试时从同一 VLA 采样多个候选动作块，以轻量 Relative Action
Critic（RAC）做成对偏好比较，再通过 tournament 选出一个候选。它与当前 ETSF 的部署形态最接近：
都不要求候选 critic 自己生成动作，最终效果都必须由冻结 actor 的闭环成功率验证。

可以迁移的部分：

- 以相对候选偏好替代不稳定的绝对 value 标度；
- 固定同一候选池，以 `N=1/4/8` 比较测试时计算量与成功率；
- 使用候选动作、候选差分、当前 proprio/state 和共享视觉语境作为公平输入；
- 把 pairwise preference accuracy 与闭环 `ΔSR` 分开报告。

不能直接迁移的部分：

- RAC 没有显式下一事件、竞争风险持续时间、失败/恢复和对象状态变化监督；
- 论文在 LIBERO-LONG 上验证 PI0.5，不能把其失败率降幅横向搬到 RoboTwin 五本体；
- tournament 的非传递偏好可能依赖 bracket 顺序，必须冻结 pairing/bracket 或改用全 pair 聚合；
- 若更换 actor、候选采样温度或视觉 backbone，结果就不能归因于 critic 结构。

因此 VLA-ATTC 应落实为 B4 的强版本：用与 ETSF 完全相同的 branch 监督、容量上限和候选集合，
训练一个 action-pair RAC：分别编码 `a_i`、`a_j`、`a_i-a_j` 与当前 canonical state，以 focal
loss 学习真实 branch 偏好，并冻结 tournament bracket；正式 nested runner 同时报
`RAC@4/RAC@8` 与 `ETSF@4/ETSF@8`。这能直接
检验事件结构化多任务监督是否优于纯相对排序，而不是只和弱标量 MLP 比较。

### 6.6 WCM / World Value Model

[WCM][wcm-paper]指出单帧 value 回归在部分可观测控制中缺少时序状态，并以轻量未来 latent
预测目标辅助 critic；[World Value Model][wvm-paper]同样把世界模型的时序表征用于通用机器人
value estimation。两者支持“价值预测需要世界建模监督”这一大方向，但并不等于当前事件模型。

可以迁移的部分：

- 在同一历史窗口上增加 future-latent predictive loss，作为 B5 的强实现；
- 共享 encoder 后同时优化 proper value/ranking 与未来状态预测；
- 对比 single-frame scalar、future-latent critic 和事件结构化 critic；
- 同时报离线 value-order/prediction 指标和冻结 actor 闭环成功率。

不能直接迁移的部分：

- WCM 主要服务 VLA RL post-training，当前实验是 critic-only、actor 不更新；
- latent prediction 正确不保证候选动作改变了任务事件，也不自动产生跨本体规范语义；
- 其视觉 latent、backbone 和训练数据合同与 RoboTwin EE16 branch 数据不同；
- 直接加载权重会同时改变表征、数据规模与容量，无法成为结构消融。

公平比较应在相同 root branch 上训练 matched-capacity future-latent baseline：用 action-conditioned
residual predictor 预测下一状态 latent，联合 value loss、latent MSE 与防坍塌正则，并保持 actor、候选、
rollout 数和 LOBO split 不变。ETSF 的可主张创新不应写成“首次给 critic 加世界模型”，而应写成：
**把候选动作后果分解为跨本体规范事件、竞争风险持续时间、恢复与对象效果，并证明这种可解释的
事件因子化在严格 held-out-body、critic-only best-of-N 中比标量 RAC 和非结构化 future-latent
critic 得到更好的校准、oracle regret 与配对任务成功率。**

## 7. 正式 100 seeds/cell 配对协议

### 7.1 配对单位

配对 key 是：

```text
(heldout_body, condition, requested_seed)
```

同一 key 下：

- `actor_baseline` 固定执行 ordered candidate set 的 index 0；
- `etsf_best_of_4` 只在完全相同的四候选中选择最高冻结分数；
- 精确同分选择最低 candidate index；
- 两方法各自从同一 resolved reset 的干净副本开始；
- 偶数 seed ordinal 执行 `actor → ETSF`，奇数反序，抵消顺序影响；
- 分叉后各自在自己的闭环状态继续查询，不能读取另一路未来；
- 二值成功只能来自 native simulator task checker。

### 7.2 成功率定义

令 `Y(b,c,s,m) ∈ {0,1}` 表示 body `b`、condition `c`、seed `s`、方法 `m` 的完整
任务成功。每个 cell 的成功率为：

```text
SR(b,c,m) = mean_s Y(b,c,s,m)
ΔSR(b,c)  = mean_s [Y(b,c,s,ETSF) - Y(b,c,s,actor)]
```

主结果不是把 1000 行当成一个无结构比例，而是按以下层级完整报告：

1. 10 个 `body × condition` cell；
2. 5 个 body 的 equal-condition macro；
3. 2 个 condition 的 equal-body macro；
4. global equal-body-condition macro。

### 7.3 Macro Cross-Emb

参照 MOTIF 的“机器人等权 macro”思想，但按当前严格 LOBO 重新定义：

```text
CrossEmb-SR(m) = (1 / 5) Σ_b [(1 / 2) Σ_c SR(b,c,m)]
CrossEmb-ΔSR   = (1 / 5) Σ_b [(1 / 2) Σ_c ΔSR(b,c)]
```

由于当前每个 cell 都恰好有同一组 100 seeds，global pooled mean 与 equal-cell macro 的点估计
数值相等；仍必须叫 equal-cell macro，因为未来 cell 缺失或五任务扩展后不能依赖这个巧合。

“Cross-Emb”只有在每个 `b` 都是对应 fold 真正未见的 body 时才成立。如果五个 body 都参与了
同一个 checkpoint 的训练，这个数只能叫 multi-body performance，不能叫 held-out transfer。

## 8. 置信区间、McNemar 与成功 gate

### 8.1 单方法 SR 区间

每个 cell 对 actor SR 和 ETSF SR 报 Wilson 95% CI；可以补充 equal-tailed
Clopper–Pearson，但不使用 `mean ± standard deviation` 代替二项比例区间。

### 8.2 配对 ΔSR 区间

`ΔSR` 的主区间使用 20,000 次 paired requested-seed cluster percentile bootstrap。重采样单位
是 requested seed：

- 单 cell：每个 seed cluster 含该 cell 的一对结果；
- body macro：同 seed 的 clean/randomized 两个 cell 一起抽；
- condition macro：同 seed 的五个 body 一起抽；
- global macro：同 seed 的五 body × 两 condition 共十个 cell 一起抽。

这样不会把相同 seed 在多个 body/condition 的重复使用错当成 1000 个独立 IID 样本。
percentile bootstrap 不是 exact CI，报告名称中应保留 `not_exact`。

### 8.3 Exact McNemar

对每个配对报告：

- `b`：actor 成功、ETSF 失败；
- `c`：actor 失败、ETSF 成功；
- `n00` 与 `n11`；
- discordant pairs 上的 two-sided exact binomial McNemar p-value。

若 `b+c=0`，规定 `p=1`。不能只报两条独立 SR 的显著性检验。global pooled exact McNemar
没有建模 repeated-seed cluster，必须明确标成组合描述；global 的区间推断仍以 seed-cluster
bootstrap 为主。

### 8.4 预注册 improvement gate

至少同时满足：

1. global macro `CrossEmb-ΔSR` 的 cluster-bootstrap 95% CI 下界严格大于 0；
2. global exact McNemar `p < 0.05`；
3. 每个 held-out-body macro 的 `ΔSR` 点估计非负；
4. clean 与 randomized 两个 condition macro 的 `ΔSR` 点估计均非负；
5. 1000 pairs 完整，无 outcome 后删行、补 seed、换 checkpoint 或改 N。

如果只满足点估计为正而 CI 穿过 0，应写“趋势为正但证据不足”，不能写“显著提高”。

## 9. 阶段进度不是替代成功率

当前 `move_can_pot` 的事件链是：

```text
e0 → e12 → e3 → e4 → eK
0     .25    .50   .75   1.0
```

- `e0`：默认/尚未形成可归类后果；
- `e12`：can 已移动或抬起；
- `e3`：can 已进入官方成功位置区域；
- `e4`：满足位置、姿态和高度，仅缺最终 gripper-open 条件；
- `eK`：native simulator checker 判定完整成功。

每个方法报告 mean terminal max-event progress、paired `Δstage`、seed-cluster 95% CI，以及每个
阶段的 reach rate。它可以解释模型把失败推进到了哪一步，尤其适用于总体 SR 较低时；但
`e4=0.75` 仍是失败，不能把阶段改善改写成完整任务成功。

这种分阶段报告与 LAP 的 staged progress 思路一致，但当前事件定义是 `move_can_pot` 专用规范，
不是 LAP 或 RoboTwin 对所有任务统一给出的事件本体。

## 10. Oracle headroom 与 oracle regret

对同一 decision group `g` 的 N 个真实候选分支，令：

- `y(g,i)` 为完整分支二值成功；
- `p(g,i)` 为 terminal stage progress；
- `q(g,i)` 为连续 goal progress；
- `i*` 为 critic 选择的候选；
- `i0=0` 为 actor baseline 候选。

报告以下量：

```text
success headroom(g) = max_i y(g,i) - y(g,i0)
success regret(g)   = max_i y(g,i) - y(g,i*)
stage regret(g)     = max_i p(g,i) - p(g,i*)
goal regret(g)      = max_i q(g,i) - q(g,i*)
```

并按 `heldout body × condition` 先求均值，再做五本体等权 macro。还要报告 mixed-success group
占比、mixed-success selection accuracy、pairwise accuracy，以及 uncertainty/coverage 下的
selected-failure 与 oracle-regret AURC。

解释规则：

- oracle SR 高、selected SR 低：critic 排序仍需改善；
- oracle SR 与 baseline SR 相同：候选池没有 success headroom，继续堆 critic 头不会提高 SR；
- stage oracle 有差距而 success oracle 无差距：候选只改善局部进度，不能完成任务；
- N=8 比 N=4 oracle 高而 selected 不高：扩大候选池有效，但 critic 没捕获新增候选。

重要边界：当前正式 `actor vs ETSF` 两条闭环 rollout 本身不能产生 oracle regret，因为未执行的候选
没有真实未来。oracle 必须来自单独冻结的同根候选 branch 数据，或额外执行全部候选的密封诊断；
它永远不能参与正式在线选动作，也不能用 simulator lookahead 冒充可部署结果。

## 11. Critic 准确预测指标

预测指标按 held-out body 计算后再等权宏平均，并始终带 support/mask：

<!-- markdownlint-disable MD013 -->

| 输出 | 主诊断 | 必须避免的错误 |
| --- | --- | --- |
| post/next event | macro-F1、NLL、accuracy、混淆矩阵 | 只报多数类 accuracy |
| success | Brier、NLL、ECE、PR-AUC；双类时 AUROC | 单类别 AUROC 写成 0 或 1 |
| holding time | observed MAE、density NLL、censored survival NLL | 丢弃右删失样本 |
| object effect | raw-meter translation/rotation MAE、Student-t mixture NLL | 用归一化空间误差冒充物理误差 |
| recovery/regress | support、PR 曲线、AP/F1 | 没有正例仍声称可预测 recovery |
| uncertainty | error-risk coverage、AURC、harmful-rate vs coverage | 把 ensemble entropy 直接叫“已校准” |
| ranking | mixed selection accuracy、pairwise accuracy、NDCG、oracle regret | 用逐行 AUC 代替 decision-group 排序 |

<!-- markdownlint-enable MD013 -->

这些指标证明的是“为什么重排可能有效/为什么失败”，主张“提高任务成功率”仍只能由第 7--8 节的
正式 paired `ΔSR` 证明。

## 12. N=4/N=8 与候选池效果

候选数实验必须使用 nested pool：`N4 == N8[:4]`。对同一新 seed roster 配对执行
`candidate-0`、`best-of-4` 和 `best-of-8`，并报告：

- `ΔSR(N4−actor)`、`ΔSR(N8−actor)`、`ΔSR(N8−N4)`；
- 对应 `Δstage`；
- oracle SR/stage 随 N 的增量；
- success/stage/goal oracle regret；
- latency、GPU 显存与每 query wall time。

如果 N=8 只增加 oracle headroom，却没有提高 selected SR，应优先改排序监督；如果 oracle 也不变，
应改候选生成的多样性，而不是继续扩大共享头。

## 13. 从单任务走向五任务的下一步

### 13.1 建议的五任务集合

RoboTwin2 官方任务页列出 50 个双臂任务并支持五种本体。第一阶段建议用以下五个具有不同事件
结构的任务，而不是选五个几乎相同的 pick-and-place：

<!-- markdownlint-disable MD013 -->

| task | 主要事件结构 | 选择理由 |
| --- | --- | --- |
| [`move_can_pot`][move-can-pot] | grasp → lift → transport → align → release | 保留当前已实现基准 |
| [`lift_pot`][lift-pot] | 双臂接触/抓持 → 同步 lift | 测试无 release 的持续协同 |
| [`press_stapler`][press-stapler] | reach → contact → press/activate | 测试非搬运、接触触发事件 |
| [`open_microwave`][open-microwave] | grasp handle → articulated motion → open | 测试关节对象与长时保持 |
| [`stack_bowls_two`][stack-bowls-two] | 双对象抓取 → 对齐 → stack → release | 测试多对象、顺序和精确放置 |

<!-- markdownlint-enable MD013 -->

任务页证明的是 simulator/五本体支持与官方生成统计，不证明本地 archive 已完整下载，也不证明每个
body 都有足够的正负 critic 分支。正式加入前必须逐任务审计 immutable revision、archive SHA、
member count、schema、成功 checker 和实际 label support。

### 13.2 事件建模必须从“固定五类语义”升级

`move_can_pot` 的 `e3/e4` 不能原样解释 microwave 或 stapler。五任务模型应拆成：

- 共享原子谓词/效果：`reach`、`contact`、`grasp`、`lift`、`transport`、`align`、
  `release`、`articulate`、`activate`、对象 SE(3)/joint-state effect；
- task-specific event graph：每个任务冻结自己的合法偏序、success checker 和 applicability mask；
- 共享 event/effect core：学习动作对这些原子后果、holding time 和不确定性的影响；
- 小型 task adapter/graph decoder：把共享后果映射成任务专用的 `e_t → e_{t+1}`。

这样才能声称“共享事件机制跨任务”，而不是把五个任务强行压进同一套位置阈值。

### 13.3 五任务正式评估规模

第一阶段仍做每任务五折 body-LOBO：训练时完整留出一个 body，但共享 core 可以读取其余四个 body
的五任务训练数据。正式评估规模为：

```text
5 tasks × 5 held-out bodies × 2 conditions × 100 paired seeds
= 5000 pairs
= 10000 actor/ETSF closed-loop rollouts
```

宏平均采用严格层级：先在每个 `task × body × condition` cell 内算配对差，再 condition 等权、
body 等权、task 等权。全局 bootstrap 仍按 requested seed 聚类，一次抽样带走该 seed 的全部
`5 tasks × 5 bodies × 2 conditions` cell。

为了证明跨任务，再增加两个实验轴：

1. **LOTO**：留出一个任务，训练另外四任务，目标任务只允许冻结 core 零样本或固定 K-shot
   task adapter；
2. **double holdout**：同时留出一个 task 与一个 body，测试未见任务—本体组合。

最终应分开报告：

- body transfer：seen task / unseen body；
- task transfer：unseen task / seen body；
- joint transfer：unseen task / unseen body；
- multi-task in-distribution：seen task / seen body。

四者不能合成一个含糊的“通用迁移率”。

## 14. 实验完成顺序

按信息价值和成本排序，下一步应该是：

1. 完成当前 `move_can_pot` 的五折 LOBO 训练闭包；
2. 完成 100 seeds/cell 的 actor vs ETSF 正式配对 `ΔSR`；
3. 同步报告 stage、mixed selection、oracle headroom/regret，判断瓶颈在候选池还是 critic；
4. 跑 B1--B7，同数据、同候选、同预算证明结构贡献和迁移贡献；
5. 跑 nested N4/N8，只有 oracle headroom 随 N 增长时才继续扩候选池；
6. 对五任务做官方 archive 与 checker 审计，并逐任务冻结 event graph；
7. 先做五任务 body-LOBO，再做 LOTO 和 double holdout；
8. 所有确认集只打开一次，结果完整报告，不按结果删 body、task、condition 或 seed。

在第 2 步通过前，只能写“共享事件头具备跨本体接口并正在验证”；在第 7 步通过前，只能写
“单任务跨本体”，不能写“通用跨本体事件世界模型”。

## 15. 与仓库现有实现的对应关系

- 五折预注册：
  `scripts/preregister_robotwin2_move_can_pot_five_body_lobo_v1.py`；
- 五折共享头训练：
  `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`；
- 正式 100 seeds/cell 执行：
  `scripts/run_robotwin2_five_body_paired_success_v1.py`；
- 严格结果统计：
  `scripts/evaluate_robotwin2_cross_embodiment_paired_success_v1.py`；
- nested N4/N8：
  `scripts/run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py`；
- 当前任务事件规范：
  `scripts/robotwin2_move_can_pot_analytic_event_spec_v2.py`；
- success-only/full 离线消融：
  `scripts/run_robotwin2_five_body_lobo_offline_ablation_v1.py`。

上述文件能证明协议和统计实现存在，不能单独证明模型已准确预测或已经提高成功率。最终证据必须
同时包含五折训练 receipt、冻结 checkpoint SHA、完整 1000-pair rollout provenance、正式报告和
独立一致性验证。

## 16. 官方来源

- [RoboTwin2 官方项目与文档][robotwin-doc]
- [RoboTwin2 50 个任务列表][robotwin-tasks]
- [RoboTwin2 官方数据集][robotwin-data]
- [Open X-Embodiment 项目页][oxe-project]
- [Open X-Embodiment / RT-X 论文][oxe-paper]
- [Open X-Embodiment 官方代码][oxe-code]
- [RoboCritic 项目页][robocritic-project]
- [RoboCritic 论文][robocritic-paper]
- [MOTIF 论文][motif-paper]与[官方代码][motif-code]
- [LAP 论文][lap-paper]与[官方代码][lap-code]
- [MimicGen 项目页][mimicgen-project]、[论文][mimicgen-paper]与[官方代码][mimicgen-code]
- [VLAC 论文][vlac-paper]与[官方代码][vlac-code]
- [ProgressVLA 论文][progressvla-paper]
- [VLA-ATTC 论文][vla-attc-paper]
- [WCM 论文][wcm-paper]
- [World Value Model 论文][wvm-paper]

[robotwin-doc]: https://robotwin-platform.github.io/doc/
[robotwin-tasks]: https://robotwin-platform.github.io/doc/tasks/
[robotwin-data]: https://huggingface.co/datasets/TianxingChen/RoboTwin2.0
[move-can-pot]: https://robotwin-platform.github.io/doc/tasks/move_can_pot.html
[lift-pot]: https://robotwin-platform.github.io/doc/tasks/lift_pot.html
[press-stapler]: https://robotwin-platform.github.io/doc/tasks/press_stapler.html
[open-microwave]: https://robotwin-platform.github.io/doc/tasks/open_microwave.html
[stack-bowls-two]: https://robotwin-platform.github.io/doc/tasks/stack_bowls_two.html
[oxe-project]: https://robotic-transformer-x.github.io/
[oxe-paper]: https://robotics-transformer-x.github.io/paper.pdf
[oxe-code]: https://github.com/google-deepmind/open_x_embodiment
[robocritic-project]: https://robocritic.cs.columbia.edu/
[robocritic-paper]: https://arxiv.org/abs/2606.21572
[motif-paper]: https://arxiv.org/abs/2602.13764
[motif-code]: https://github.com/buduz/MOTIF
[lap-paper]: https://arxiv.org/abs/2602.10556
[lap-code]: https://github.com/lihzha/lap
[mimicgen-project]: https://mimicgen.github.io/
[mimicgen-paper]: https://proceedings.mlr.press/v229/mandlekar23a.html
[mimicgen-code]: https://github.com/NVlabs/mimicgen
[vlac-paper]: https://arxiv.org/abs/2509.15937
[vlac-code]: https://github.com/InternRobotics/VLAC
[progressvla-paper]: https://arxiv.org/abs/2603.27670
[vla-attc-paper]: https://arxiv.org/abs/2605.01194
[wcm-paper]: https://arxiv.org/abs/2607.29613
[wvm-paper]: https://arxiv.org/abs/2606.24742
