# ETSF 事件世界模型：与 VLAC / ProgressVLA 的创新边界与基线审计

> 审计日期：2026-08-27。本文只对用户指定的两篇工作做逐项原文核对，不构成对全部
> 事件世界模型、SMDP、机器人 reward model 或 VLA planning 文献的系统检索。因此可以写
> “区别于 VLAC/ProgressVLA”，在完成更广泛检索前不能写 “首次（first）” 或 “唯一”。

## 1. 论文原文事实

### 1.1 VLAC（arXiv:2509.15937v1）

- VLAC critic 的输入是两幅**已经观测到的**图像和语言目标，输出有符号的标量进度差；
  另有单帧 done 判断。进度标签来自成功、高效轨迹的时间顺序，具体为
  `Δt/(T-i)`；倒序对、静止帧和错配任务描述构成负样本
  （[§3.1.1, Eq. 1–4](https://arxiv.org/html/2509.15937#S3.SS1.SSS1)）。
- critic 的进度任务不要求动作，因此能够混用人类与不同机器人数据；论文明确把这点作为
  绕开异构动作空间的手段，并通过 reference trajectory 做 one-shot in-context progress
  transfer（[§3.1.1](https://arxiv.org/html/2509.15937#S3.SS1.SSS1)）。
- 同一个 2B/8B 自回归模型还可以生成 delta-EEF 数字动作，但 critic 不是
  `p(outcome | current state, candidate action)`：动作生成和 pairwise progress 是用 prompt
  区分的联合任务（[§3.1.2](https://arxiv.org/html/2509.15937#S3.SS1.SSS2)）。
- 训练规模远大于当前 ETSF：论文报告 3000+ 小时人类数据、1200 小时公开机器人数据、
  15+ 小时自采数据和约 4000 万训练样本
  （[§3.1](https://arxiv.org/html/2509.15937#S3.SS1)、
  [§3.1.3](https://arxiv.org/html/2509.15937#S3.SS1.SSS3)）。
- VLAC 已经报告跨实体数据集上的进度评估：例如未在训练集内的 RT1 在 one-shot 下
  VOC-F1 为 0.95；也评估了 Dobb-E、RH20T 和人手 EgoDex。因此“进度语义可以跨机器人”
  不能作为 ETSF 独有主张（[§4.4](https://arxiv.org/html/2509.15937#S4.SS4)）。
- 它把进度 delta 和 done 用作真实机器人 PPO 的 reward/termination，并报告四项任务的
  在线改善；还在同类机器人数量 `1/2/4/8` 上做共享训练扩展
  （[§3.2.2](https://arxiv.org/html/2509.15937#S3.SS2.SSS2)、
  [§4.6](https://arxiv.org/html/2509.15937#S4.SS6)、
  [§4.7](https://arxiv.org/html/2509.15937#S4.SS7)）。这不是“冻结 critic 跨本体
  迁移”的严格因果实验，但足以排除“共享模型用于多机器人训练”这一宽泛创新表述。

### 1.2 ProgressVLA（arXiv:2603.27670v1）

- progress estimator 为 `P(language, initial observation, current observation)`，输出
  `[0,1]` 标量，以专家轨迹的 `t/T` 为监督，依赖“成功专家轨迹近似单调”的假设
  （[§IV-A, Eq. 1–2](https://arxiv.org/html/2603.27670#S4.SS1)）。
- 它已经是动作条件世界模型：inverse encoder 把 `(o_t,o_{t+N})` 压成 latent action，
  forward decoder 用 `(o_t,a^z)` 预测未来视觉特征；随后联合优化 world/progress/joint loss
  （[§IV-B–C, Eq. 3–8](https://arxiv.org/html/2603.27670#S4.SS2)）。
- 其控制接口不是事后简单打分。latent-action diffusion 的每一步去噪都通过可微 world model
  把候选 latent action 映射到未来状态，再反传 progress gradient 做 classifier guidance
  （[§IV-D, Eq. 9–11](https://arxiv.org/html/2603.27670#S4.SS4)）。在线阶段还用
  KL-constrained progress improvement 微调 denoiser
  （[§IV-E, Eq. 12–20](https://arxiv.org/html/2603.27670#S4.SS5)）。
- 它明确提出“embodiment-agnostic latent action expert + embodiment-specific action decoder”的
  two-stage 设计，所以“高层共享、低层动作解码适配本体”也不是 ETSF 单独的新意
  （[§IV-D](https://arxiv.org/html/2603.27670#S4.SS4)）。
- 论文已用 `w/o classifier guidance` 做直接消融：CALVIN 的 success 指标
  `92.7→93.6`，LIBERO 平均成功率 `81.5→83.3`；真实 ARX 平台五任务平均
  `66→76`，每任务 20 次测试
  （[Table I](https://arxiv.org/html/2603.27670#S2.T1)、
  [§V-D](https://arxiv.org/html/2603.27670#S5.SS4)、
  [§V-E](https://arxiv.org/html/2603.27670#S5.SS5)）。因此“用世界模型改变动作候选能提高
  成功率”已经有非常相近的先例。
- 论文的迁移证据主要是 CALVIN 场景、LIBERO 任务和真实机器人上的 lighting/novel-object
  shift；真实控制只使用一种 ARX X5 本体。论文称其结构具有 cross-embodiment flexibility，
  但没有报告“同策略换本体、冻结共享核心、只训练小 adapter”的 leave-one-embodiment
  成功率实验（[§V-C–F](https://arxiv.org/html/2603.27670#S5)）。这恰好给 ETSF 留下一个
  **必须靠严格实验而不是措辞**建立的差异。

## 2. 逐项重合与可守住的差异

| 维度 | VLAC | ProgressVLA | 当前 ETSF | 审计判断 |
|---|---|---|---|---|
| 任务进度理解 | 两帧 + 语言 → signed progress | 初始/当前帧 + 语言 → normalized progress | event/value heads | 高度重合；不能以“有进度 critic”为创新 |
| 动作条件预测 | critic 本身不看候选动作 | latent action → future visual latent → progress | native action chunk → event/time/object/success | 与 ProgressVLA 高度重合；不能以“动作条件世界模型”为创新 |
| 失败、停滞、回退 | 构造负样本，识别 regression/stagnation | online buffer 含 recovery/near-failure，并施加单调约束 | relative `stay/advance/skip/regress` + predicates | “能识别回退”也不新；差异在显式离散转移及其独立评测 |
| 时间 | 标量相对进度和 done | `t/T` progress、steps/termination | event-conditioned `log1p(D)` 分布 + 右删失生存似然 | 这是相对两篇论文最清楚的结构差异 |
| 对象效果 | 借助 VLM 感知但不输出对象变化分布 | 预测未来视觉特征 | 对象位姿 delta 均值/尺度 + post predicates | 输出接口不同；必须证明比 future-latent/progress 基线更准或更有用 |
| 不确定性与安全 | 未给出 ensemble abstention guard | 未给出候选替换 abstention guard | aleatoric + ensemble epistemic + margin/distance/calibration fallback | 可形成系统贡献，但需 calibration/AURC 和 harmful-rate 证据 |
| 策略接入 | actor/critic 同一自回归模型，PPO 特定 | 可微 diffusion classifier guidance | actor 外部候选重排，adapter 接原生候选 | “非可微、冻结 actor、可回退”是清楚的工程/算法差异 |
| 跨本体 | action-free progress 跨实体数据评估；多机器人共享训练 | latent/action-decoder 因子化，宣称 flexibility | State/Policy/Body/Clock/Predicate adapters + 冻结核心审计 | 只能称 transfer-ready；严格可归因实验通过后才是实证差异 |
| 监督 | 海量时间排序 + 构造负样本 | OXE 专家 `t/T` + online success/monotonicity | 同初态 simulator action branch 的事件/对象/时钟/终局标签 | 受控干预监督是差异；数据规模和任务多样性目前明显更弱 |

相对这两篇论文，最稳妥的定位不是“另一个 progress-guided world model”，而是：

> **A policy-external, action-conditioned structured event-and-holding-time model with
> calibrated abstention for native VLA candidate selection.**

中文建议为：

> **面向原生 VLA 候选选择的策略外置、动作条件结构化事件—持有时间模型，并以校准不确定性
> 实现安全拒绝。**

这里刻意不在名称中先写“跨本体”，因为迁移目前是接口和协议属性，不是已完成的结果。
迁移实验通过后可追加 “with factorized few-shot policy/embodiment transfer”。

## 3. 当前绝对不能过度声称的内容

1. **不能声称首次把世界模型用于动作选择。** ProgressVLA 已通过未来 latent 和 progress
   gradient 引导动作生成。
2. **不能声称首次让 progress/event knowledge 跨本体。** VLAC 已做跨实体 progress 评估；
   ProgressVLA 已提出 embodiment-agnostic latent action 与 action decoder 分解。
3. **不能声称当前已经跨本体提高成功率。** 现有 `OpenVLA+Piper` 与
   `SmolVLA+Aloha` 同时改变 policy、body、state hook、动作 chunk 和 clock，不能归因。
   `docs/ETSF_strict_policy_body_transfer_protocol.md` 已正确要求分别做同 body 的 P-transfer
   与同 policy 的 E-transfer。
4. **不能声称 recovery 已准确建模。** 当前正式 config 的
   `recovery_supervised=false`；第三类输出存在并不等于有正样本或可评测。现在能报告的是
   `regress` transition（且要给 support/F1），不是 `p(recovery)`。
5. **不能把五个 predicate 全称为“可逆谓词”。** 当前 `lifted/near_goal/stationary` 可反转，
   `moved` 是累计谓词，`success` 只在成功终帧置真。准确表述是“动态原子谓词，其中三个
   显式可逆”。
6. **不能在 OOF/fresh 指标完成前称准确预测或提高成功率。** 代码存在、loss 下降或 oracle
   headroom 都不证明 held-out accuracy，也不证明 selector 找到了 oracle 候选。
7. **不能把 ensemble entropy 直接称为校准不确定性。** 必须报告 success Brier/ECE、
   uncertainty-error correlation、risk-coverage/AURC，并验证高不确定样本被 guard 拒绝后
   harmful rate 下降。
8. **不能把“可插拔”写成“已适配任意算法”。** 目前可证明的是接口对 OpenVLA 可用；
   SmolVLA/ACT/diffusion 每一种都要有原生 state/action adapter、校准契约和闭环结果。
9. **谨慎使用“因果/反事实”。** 同 requested/resolved seed、同语言、同初始 simulator
   state 下实际执行不同首动作，是受控 action intervention；但当前结果不等于识别了可跨
   环境外推的因果机制。建议写 “controlled same-state intervention branches”，不要写
   “learns the true causal dynamics”。
10. **谨慎使用“完整 semi-Markov world model”。** 当前代码实现的是事件条件持有时间分布、
    右删失似然和 duration-aware discount；在多步 event rollout、Markov sufficiency 与
    calibrated transition kernel 未验证前，建议写 “event semi-Markov component/model”，
    不写“完整生成式 SMDP 模拟器”。
11. **不能声称跨任务语言泛化。** 当前正式 OpenVLA 数据集中在一个 RoboTwin 任务；VLAC
    和 ProgressVLA 使用大规模多任务预训练。单任务结构化准确率不能外推为 open-vocabulary
    event understanding。

## 4. Baseline 能否直接迁移

### VLAC

官方提供了 [VLAC 模型与推理代码](https://github.com/InternRobotics/VLAC)，因此可以把冻结
VLAC 当作**外部 pairwise progress/done evaluator**，在相同录像和语言上报告 VOC、VROC、
done accuracy，或把其 progress 作为候选执行后的离线 oracle-style 诊断。它不能直接替换
ETSF 的事件世界模型权重：输入是 RGB pair 而非 OpenVLA hidden + 未执行 action，输出也没有
next-event、删失 duration、object delta 或 uncertainty heads。官方仓库把训练指向 InternVL2，
没有提供与 ETSF schema-v5 对齐的完整训练流水线，所以不能把轻量同数据 regressor 称为
“VLAC 复现”。

### ProgressVLA

论文原文没有给出官方代码链接；其完整方法还依赖 UniVLA 式视觉 world model、latent action
expert、action decoder、noise-conditioned evaluator 与 diffusion denoiser。OpenVLA 的离散
自回归动作 token 无法直接接入其 classifier-gradient 路径。因此当前最公平可执行的是仓库中
已有的两个**同数据方法级 baseline**：

- `direct`: `state + candidate action -> scalar progress`；
- `latent_future`: `state + candidate action -> predicted future latent -> scalar progress`。

入口见 `scripts/train_openvla_etsf_progress_baseline.py` 与
`docs/ETSF_progress_baseline.md`。论文中必须称其为 “VLAC/ProgressVLA-style lightweight
same-data baselines”，不能称官方复现。若论文主张需要与 ProgressVLA 的绝对 SOTA 数值比较，
则应在其 CALVIN/LIBERO diffusion setting 上重建完整 pipeline，而不是把 RoboTwin 上的轻量
baseline 数值与论文表格横向相减。

## 5. 必须完成的实验矩阵

### 5.1 同数据、同候选的结构贡献

在完全相同的 train/OOF group 和 candidate set 上比较：

| Baseline / 消融 | 要回答的问题 |
|---|---|
| actor deterministic candidate | 插件是否真正改善冻结 actor |
| candidate oracle | 候选生成器的可提升上限；不是可部署结果 |
| action-conditioned success-Q | 结构化监督是否优于只学终局成败 |
| direct scalar progress | 是否只是标量 progress critic |
| action → future latent → scalar progress | 是否只是 ProgressVLA 式预测后打分 |
| flat absolute next-event | relative transition/predicate 是否必要 |
| full minus duration/object/uncertainty/action/history | 各结构头的增量价值 |

预测至少报告：next-event/relative-transition macro-F1 与 NLL、predicate macro-F1、observed
duration MAE/NLL、right-censored survival NLL、object-delta raw-meter MAE/NLL、success
Brier/ECE/PR-AUC、within-group pair accuracy、NDCG 和 uncertainty AURC。每一项必须胜过它的
朴素基线，而不是只报告绝对数字。

### 5.2 闭环成功率

- validation 只冻结 temperature、score weights、distance/margin/uncertainty guard；
- fresh confirmation 只打开一次；按同 seed 报 baseline/selected/oracle、helpful、harmful、
  changed coverage、paired delta、bootstrap 95% CI 与 McNemar/sign test；
- 成功率点估计 `>0` 且 95% CI 下界 `>=0`，才写“提高成功率”；否则写 monitor-only 或
  inconclusive。

这一步是相对 ProgressVLA 的最低对照要求：它已经报告 `w/o guidance` 与 `w/ guidance` 的
闭环成功率，ETSF 不能只用离线 AUC 回答。

### 5.3 非特权输入、事件性与任务广度

- 主闭环结果不能依赖 simulator object pose 直接给出当前 event/predicates；应同时报告
  privileged-oracle observer 与由 RGB/actor hidden 训练的 observer，并把后者的 event F1、
  predicate F1、校准覆盖率和闭环成功率作为可部署结果。否则与直接从 RGB 学习的 VLAC /
  ProgressVLA 不构成公平系统比较。
- 至少扩展到多个具有不同事件偏序和失败模式的任务，并做 leave-one-task-out 或
  shared-event / task-specific-event 对照。当前单一 `move_can_pot` 只能证明一个任务上的
  schema，不能证明通用“事件建模”；两篇相似工作都使用多任务大规模预训练和多任务控制
  评测（[VLAC §3.1/§4](https://arxiv.org/html/2509.15937#S3.SS1)、
  [ProgressVLA §V](https://arxiv.org/html/2603.27670#S5)）。
- 若论文要写 recovery，需先冻结操作定义（例如 regress 后在同一 episode 再 advance 且最终
  success），采集足够正例，再令 `recovery_supervised=true`；报告 recovery PR-AUC/F1 和
  相对仅用 regress 标签的增量。没有这一步就删除 recovery 主张。

### 5.4 真正可归因的迁移

严格按 `docs/ETSF_strict_policy_body_transfer_protocol.md` 分开：

- P-transfer：同 task、同 body，只换 policy；
- E-transfer：同 task、同 policy，只换 body；
- 每条轴报告 `N={0,5,10,20,50}`，比较 frozen shared core + adapter、同 N/同容量
  target-from-scratch、no-factorization 与 full-finetune upper bound；
- shared core 除预留 embedding 行外 bit-exact，目标 validation 冻结 guard，目标
  confirmation 每任务至少 50 groups且只打开一次；
- 同 N 下准确率/成功率优于 target-from-scratch，并以更少可训练参数达到，才可以写
  “sample-efficient transfer”。

这是最可能超出两篇相似工作的实验贡献：并非“架构上看起来能跨本体”，而是把 policy 与
body 两个迁移轴拆开、审计冻结权重，并证明小 adapter 在确认集上的预测和控制收益。

## 6. 建议论文贡献表述

### 6.1 现在即可写（架构/协议事实，不包含效果）

1. “We formulate native VLA candidate evaluation as a structured event-effect prediction
   problem, jointly modeling relative event transitions, post-action predicates,
   event-conditioned holding time under right censoring, object-state change, and terminal
   outcome.”
2. “Unlike actor-integrated progress guidance, our policy-external selector consumes native
   action chunks and can abstain to the frozen actor using calibrated margin, action-distance,
   and ensemble-uncertainty contracts.”
3. “We define a falsifiable transfer protocol that separates policy transfer from embodiment
   transfer and audits the shared event core for bit-exact freezing during few-shot adapter
   calibration.”

### 6.2 只有对应实验通过后才能写

- 预测门通过： “The structured model improves held-out event/time/object prediction over
  scalar-progress, latent-future-progress, and success-Q baselines.”
- fresh 门通过： “Guarded candidate selection improves paired task success by `X` points
  (`95% CI [...]`) while limiting harmful replacements to `Y%`.”
- P/E-transfer 门分别通过： “With `N` target groups, a frozen shared event core plus
  factorized adapters outperforms a matched target-from-scratch model on the held-out policy /
  embodiment.”

### 6.3 一句话创新主线

> 相比 VLAC/ProgressVLA 用单调标量进度提供 reward 或可微生成指导，ETSF 的目标是学习
> **动作干预后的结构化事件、持有时间和对象效果分布**，再以**策略外置且可拒绝的候选选择器**
> 接入异构 actor；其跨策略/本体价值由**冻结核心 + 小适配器 + 分轴密封确认**证明，而不是
> 由“共享 latent”这一架构描述推断。

这条表述既突出差异，也把“准确预测、提高成功率、跨本体、适配多算法”四个强主张绑定到
各自可验证的实验门，避免与两篇相似论文发生不必要的优先权冲突。
