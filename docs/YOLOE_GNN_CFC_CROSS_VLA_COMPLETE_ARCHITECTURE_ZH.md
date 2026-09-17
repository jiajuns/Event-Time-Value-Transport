# YOLOE + GNN 关系学习与跨 VLA 调制架构

更新日期：2026-09-17

适用分支：`gnn-cfc-event-world-model`

文档性质：实现说明、接口设计与验证边界；不是新增实验结果

> 2026-09-17 设计变更：YOLOE 实视频关系观察路径已删除 CfC。当前目标数据流是“冻结 YOLOE 候选检测与 ROI → 因果角色图 → GNN 消息传递 → 当前/历史图表示池化 → 关系、事件和目标头”。YOLOE 不生成关系真值，GNN-only 模型也必须重新训练；旧 V3/V4 CfC 指标与 checkpoint 只保留为历史对照，不能作为新模型结果。

RoboTwin 动作条件原型和已经交付的 original RGB-CfC→SmolVLA AWR 属于历史上独立的实验/交付链路，本次没有删除其代码或资产。下文涉及这些链路的 CfC 描述用于记录既有事实，不表示新的 YOLOE+GNN 观察器仍含 CfC。

## 1. 系统边界：已经有两条实现，不应混成一个已完成模型

仓库中有两条输入 ABI 不同的图关系路线：

1. **UMI YOLOE+GNN-only 视觉关系观察器**：从真实腕部 RGB 和冻结 YOLOE 检测构建二维角色图，观察已经发生的关系和事件；它不读取动作，因此是 observer，不是 `Q(s,a)`。
2. **RoboTwin 通用动作条件事件世界模型**：从仿真物理状态构建三维 typed scene graph，并读取规范化 14-D 候选动作，预测未来关系、事件、目标满足和掉落风险；它有成为 action-conditioned critic/scorer 的结构，但正式多任务训练尚未完成。

二者只共享关系 schema 和“图 → GNN → 关系/事件头”的思想；当前二维 YOLO 角色图和三维仿真 typed graph **不是同一个可直接替换的数据格式**，也不再迁移 CfC 参数。要形成完整真实机器人闭环，还需增加经过验证的“RGB/深度/位姿 → 通用 3D 图 ABI”适配层。

## 2. 端到端架构

```text
真实视频观察路径（GNN-only 代码已实现，待重新训练）

腕部 RGB_t
   │
   ├─ 全局图像 CNN ───────────────────────────────────────────┐
   │                                                          │
   └─ 冻结 YOLOE（文本 prompt，只推理）                        │
          │ boxes / role / confidence                         │
          ├─ 34-D/相机语义线索 sidecar                        │
          └─ 因果同角色跟踪 + ROI crop                         │
                    │                                         │
              二维角色图 G_t                                  │
                    │ 两层消息传递 GNN                          │
                    └─ object/target/gripper/graph embeddings ─┤
                                                               ▼
                                                   帧表示 x_t（96-D）
                                                               │
                                              last + masked-mean history pooling
                                                               │
                                                               ▼
                                                     GNN 历史表示 h_t
                                                               │
                                 ┌─────────────────────────────┼──────────────┐
                                 ▼                             ▼              ▼
                           关系/事件头                 0.3/0.6 s 预测      目标谓词头


仿真动作条件路径（代码完整，正式训练未完成）

物理状态 ──> 3D typed scene graph ──> 三层 GNN ────────────────┐
历史真实 Δt ─────────────────────────> CfC ─────────────────────┤
任务语言 + 目标关系图 ──────────────────────────────────────────┤
VLA 原生动作 ──> 本体私有运动学适配 ──> 14-D 双臂 EE chunk ──> GRU ┤
                                                               ▼
                                              动作条件未来隐状态/未来节点
                                                               │
                          ┌───────────────────┬─────────────────┼──────────────┐
                          ▼                   ▼                 ▼              ▼
                     未来关系图            事件            目标概率       成功/掉落风险
                                                               │
                                                        value / advantage
                                                               │
                                                    非负、detach 的样本权重 w
                                                               │
                          ┌────────────────────────────────────┼────────────────────┐
                          ▼                                    ▼                    ▼
                  SmolVLA flow loss                   OpenVLA token CE       pi0.5 flow loss
```

图中最后一段有两种不同用途：

- **离线后训练调制**：把已执行 chunk 的优势变成权重，对 VLA 原生损失加权；后训练完成后，部署只需新的 VLA checkpoint。
- **在线候选重排**：把 VLA 候选动作转为 14-D 规范动作，由动作世界模型评分再选最优 chunk；仓库已有 scorer 接口，但尚未验证闭环性能。

当前正式 SmolVLA Event-AWR 属于第一种。不要把第二种未来接口写成已经部署。

## 3. 核心创新性、微调成立原因与声明边界

### 3.1 真正的创新不在单个组件，而在可迁移的关系—时间—价值接口

YOLOE、消息传递 GNN 和 AWR 都不是本项目单独发明的组件。本项目更有价值的创新点，是把它们约束为一套具有明确语义边界、面向跨任务/本体/VLA 迁移的接口：

| 创新层 | 常见做法 | 本框架的处理 | 带来的价值 |
|---|---|---|---|
| 观察结构化 | 直接把整帧 embedding 输入时序网络，或用检测规则直接产生标签 | 冻结 YOLOE 只给候选实体、ROI 和几何；缺失/歧义显式保留，由下游关系模型学习 | 避免把检测启发式伪装成事件真值，同时降低小数据下从像素发现实体的难度 |
| 图关系学习 | 直接把整帧 embedding 交给黑盒时序网络 | GNN 在显式 object/target/gripper 节点和几何边上传播，再融合最后帧与历史均值 | 关系归纳偏置明确，且没有额外连续时间网络 |
| 任务表示 | 把任务名或 one-hot task id 输入模型 | 用语言与可组合的 goal-relation graph 表示目标 | 在既有 relation schema 覆盖时，新任务可通过实体关系重组表达，而不要求共享任务编号 |
| 跨本体动作 | 把不同机器人的原生关节向量直接拼接训练 | 本体私有运动学先映射为 14-D 双臂末端效果，共享模型不读取 body id/关节编号 | 将“机器人怎样实现动作”和“动作对物体关系造成什么后果”分开 |
| 价值到策略 | 把 critic embedding 或事件 token 插入某个特定策略网络 | 将 OOF value/advantage 转成非负、截断、detach 的样本权重，在原生 loss 边界调制 VLA | 同一反馈接口可覆盖 flow loss 和 token CE，部署时无需携带 critic |
| 不确定性处理 | 漏检时强行选择一个候选或发布二值成功 | missing、ambiguous、unknown、证据确认和因果稳定门分别建模 | 允许系统拒绝判断，并使误报、漏报和遮挡行为可审计 |

因此，最合适的创新概括不是“提出一种新的 GNN”，而是：

> 提出一种由冻结开放词表检测辅助构图、由 GNN 学习对象关系，并通过 loss-level advantage modulation 连接异构 VLA 的事件学习框架。

### 3.2 为什么这种 VLA 微调在优化上成立

普通行为克隆对每条动作样本等权训练：

```math
\mathcal L_{BC}=\frac{1}{N}\sum_i\ell_i^{native}
```

本框架不改变 VLA 对动作的建模方式，只根据事件回报和基线价值产生优势 `A_i`，再计算：

```math
w_i=\operatorname{clip}\left(\exp(A_i/\beta),w_{min},w_{max}\right)
```

```math
\mathcal L_{AWR}=\frac{\sum_i w_i\ell_i^{native}}{\sum_iw_i}
```

它等价于重新调整离线数据在训练目标中的有效频率：推进任务且保持安全的已执行 chunk 对梯度贡献更大，停滞、回退或掉落相关 chunk 的贡献更小。由于权重非负，VLA 仍然在拟合数据中真实执行过的动作；由于权重被截断、归一化和 `detach`，不会把策略 loss 的梯度反传到评分器，也不会仅因一个 batch 的平均权重改变而任意放大梯度尺度。

这条接口不依赖 VLA 内部 hidden size：SmolVLA 保留自己的 flow-matching，OpenVLA 类模型保留 action-token CE，pi0.5 类模型保留自己的 flow/diffusion loss。只需在 batch reduction 前保留每个样本的原生 loss，再乘同一个标量权重。因此这里的“跨 VLA”是**反馈与优化接口通用**，不是所有模型共享网络参数或动作空间。

微调完成后，权重对训练分布的偏好已经写入 VLA 参数。离线后训练路线在部署时只运行新的 VLA checkpoint，不需要在线加载 YOLOE、GNN、CfC 或 value adapter。这使关系世界模型可以作为训练期教师，而不改变真实机器人推理 ABI 和实时延迟。

### 3.3 创新成立所依赖的条件

上述方法能够执行梯度更新，并不自动等于能够提高真实成功率。要使创新主张成立，至少需要：

1. 关系/事件信号与动作后果相关，而不是只编码录像进度、任务身份或 episode 时间。
2. 数据同时覆盖成功、停滞、掉落、误抓和恢复；重加权不能学习数据中从未出现的动作。
3. value/advantage 使用 episode 级 OOF 或严格 held-out 预测，避免用同一轨迹拟合再评分造成泄漏。
4. 权重保留合理的 effective sample size，并通过上下限避免少数样本控制优化。
5. 使用同初始化、同数据和同训练步数，与普通 BC 做成对比较。
6. 在独立任务、本体和真机执行上验证，而不只比较训练 loss。

UMI GNN-only 模型是状态观察器，不读取候选动作，因此只能根据执行前后发生的事件给**已经执行的 chunk**分配信用，不能严格表示同一状态下不同动作的 `Q(s,a)`。RoboTwin 动作条件世界模型是为补足这一点而设计的，但当前仍只有 10-step smoke。新 GNN-only 模型尚未重新训练，不能声称它已经提高 VLA 或跨本体成功率。

### 3.4 论文创新声明边界

在没有完成系统性相关工作检索前，不应使用“首次”“首个”或“此前无人实现”等绝对表述。现阶段可以把创新写成**方法组合、语义接口、跨本体动作抽象、跨 VLA loss 调制和可审计因果数据契约**；最终论文的新颖性还应与 object-centric world model、graph dynamics model、continuous-time robot learning、offline AWR 和 VLA post-training 文献逐项比较。

## 4. 冻结 YOLOE 如何辅助下游学习

### 4.1 “冻结”的准确含义

YOLOE 在本项目中是**观察侧辅助检测器**：加载本地权重后设置文本类别，随后在 `torch.inference_mode()` 中运行。YOLOE 参数不加入 optimizer，不接收关系、事件或策略损失的梯度，也不被当前小数据集微调。

当前 UMI prompt 配置把开放词表检测结果映射到三个任务角色：

- `object`：操作对象，当前 prompt 为 `vegetable`；
- `target`：目标区域，当前 prompt 为 `woven bamboo tray`；
- `gripper`：执行器，当前 prompt 为 `black gripper`。

因此，冻结 YOLOE 的作用不是替系统“学会事件”，而是把像素转换为稳定、可审计的候选实体位置。任务扩展时仍需重新定义 prompt、检查召回/歧义并验证数据契约，不能仅依赖“开放词表”四个字就假定零样本可靠。

### 4.2 YOLOE 实际提供什么

每个相机、每个角色输出 8 个字段：

```text
present, ambiguous, count, confidence, cx, cy, width, height
```

再计算两组对象中心关系：`object→gripper` 与 `object→target`，每组 5 个字段：

```text
valid, dx, dy, distance, box_iou
```

所以每个相机的固定语义线索维数为：

```text
3 roles × 8 + 2 pairs × 5 = 34
```

只有某一角色恰好有一个候选时才发布其几何量。缺失和多候选歧义有单独标志，几何零值不能解释为“不接触”“掉落”或“失败”。多相机也不在这个模块中伪造跨相机三角测量。

除 34-D sidecar 外，关系图管线还使用检测框生成：

- 节点的角色、框坐标、置信度和运动特征；
- 每个节点的 RGB ROI crop，交给**可训练**的 ROI CNN；
- 节点两两之间的相对几何边。

### 4.3 冻结后，下游到底学习了什么

YOLOE 自己不学习；被训练的是其后的模块：

| 模块 | 从冻结检测中获得的条件 | 学到的内容 |
|---|---|---|
| ROI CNN | YOLOE 框裁出的节点图像 | 与任务相关的局部外观、接触边缘、目标表面线索 |
| 节点编码器 | 角色、框、置信度、轨迹年龄、框位移、歧义 | 统一节点 embedding |
| GNN | 节点 embedding 和节点对几何 | 哪些相对位置/运动组合对应抓持、支撑、进入目标区 |
| 历史融合 MLP | 最后有效图表示和掩码历史均值 | 当前关系与窗口内平均关系线索的组合 |
| 关系/事件头 | GNN 历史表示 | 当前关系、短期未来关系、阶段、转移和目标满足概率 |

这种设计在数据量有限时提供强结构先验：下游无需从头发现“哪个像素可能是物体”，而可集中学习“物体、目标和夹爪之间的关系”。代价是上游漏检或角色错配会成为系统上限，所以实现中显式保留 missing/ambiguous/unknown，而不是强行猜测。

### 4.4 YOLOE 不是自动真值标注器

可以把“批量运行 YOLOE 并生成 box/cue/graph cache”称为**自动预检测或自动预标注**，但不能把输出称为事件、关系、奖励或成功真值。当前监督来源是独立审查的事件/关系区间；`held_by_actor` 使用单独审查过的 holding 标签，其余关系行要求 `reviewed=true`。数据契约还明确记录 `yolo_used_to_generate_labels=false`。

这条边界很重要：如果直接把 `IoU>阈值` 当成“已抓住”或“放置成功”，模型只会复现检测启发式，无法学习遮挡、支撑、释放和失败恢复。

代码依据：[`semantic_cues.py`](../umi_cfc/_internal/event_rl/semantic_cues.py)、[`relational_graph.py`](../umi_cfc/_internal/event_rl/relational_graph.py)、[`relational_observer.py`](../umi_cfc/_internal/event_rl/relational_observer.py)。

## 5. GNN 如何构建和学习关系图

### 5.1 UMI 二维角色图

每个时刻最多保留 8 个节点。节点不是机器人关节，也不是固定 YOLO class id，而是带持久 track id 的角色候选。

节点 15-D 特征为：

```text
角色 one-hot(3)
+ 归一化 xyxy(4)
+ confidence(1)
+ normalized track age(1)
+ Δxyxy(4)
+ missing(1)
+ ambiguity(1)
```

有观测的节点还带一个 RGB ROI。每条有向边是 7-D：

```text
center_dx, center_dy,
log_width_ratio, log_height_ratio,
IoU, center_distance, valid
```

图的构造严格因果：

1. 对当前帧检测做角色校验和阈值过滤。
2. 仅在同角色内，以 IoU 贪心匹配已有 track；默认阈值为 `0.2`。
3. 未观测 track 最多保留 2 帧，作为短时缺失状态；新观测可淘汰陈旧缺失槽位。
4. 多候选全部保留，只有超过 8 个节点的容量时才按置信度截断。
5. `object/target/gripper` 角色只有在当前帧恰好唯一时才建立 binding；有歧义时为 `-1`。
6. 不使用未来帧插值，不使用事件标签、任务结果或动作修正跟踪。

### 5.2 单帧 GNN 消息传递

首先将 ROI CNN 的 32-D 外观特征与 15-D 节点特征拼接，映射为 48-D 节点状态 `h_i^0`。每一层对所有有效节点对计算：

```math
m_{ij}^{(l)} = \mathrm{MLP}_m^{(l)}([h_i^{(l)}, h_j^{(l)}, e_{ij}])
```

对节点 `i` 的有效邻居消息做均值聚合：

```math
\bar m_i^{(l)} = \frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)}m_{ij}^{(l)}
```

再做带残差的更新：

```math
h_i^{(l+1)} = h_i^{(l)} + \mathrm{MLP}_u^{(l)}([h_i^{(l)}, \bar m_i^{(l)}])
```

UMI GNN-only observer 使用两层共享消息传递。输出选取唯一绑定的 object、target、gripper 三个 embedding，再加全图均值池化和三个 binding-valid 标志；角色缺失时使用可学习的 missing-role embedding。它与保留 `4×4` 空间布局的全局 RGB CNN 拼接并映射到 96-D 帧表示。

GNN 学到的是**同一时刻的关系结构**：例如对象相对夹爪靠近且共同移动、对象相对目标中心进入且尺度/交叠变化。窗口内各帧的图表示通过最后有效帧与掩码均值聚合；模型不再包含 CfC、可学习时间常数或连续时间递归。

### 5.3 RoboTwin 三维 typed scene graph

仿真分支不使用 YOLO 框，而从物理引擎采集最多 8 个节点。8 类节点是：

```text
object, container, support, articulated, tool,
left_gripper, right_gripper, goal
```

每个节点 24-D，包含位姿 `xyz+quaternion`、线/角速度、可移动标志、关节物体标志与开合比例、夹爪值、相对初始位置的位移/高度变化、速度模、object/target 标志和 valid 标志。每条边 12-D，包含 3-D 相对位置、三维/平面距离、`Δz`、相对速度、两端有效性和绝对高度差。

typed GNN 先把节点类型编码成 16-D embedding，与 24-D 状态融合成 48-D 节点状态，再执行三层消息传递，最后用 mean-pool 和 max-pool 得到 96-D 场景表示。模型输入中没有机器人 ID、原生关节编号或 S1 `state25`。

这才是“多本体共享”的关键：共享模型看到的是物体关系、末端效果和目标，不看到 Aloha、Piper、ARX 等本体名称；本体差异留在图采集器和运动学动作适配器中。

代码依据：[`schema.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/universal_event/schema.py)、[`collect_robotwin.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/collect_robotwin.py)、[`model.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/universal_event/model.py)。

## 6. GNN-only 如何汇总关系历史

### 6.1 每帧先独立构图和传播

冻结 YOLOE 提供候选框、角色、置信度和 ROI。因果 tracker 生成当前帧角色图，GNN 只在当前帧有效节点之间做消息传递。每帧得到一个 96-D 图像—图融合表示 `x_t`。

### 6.2 无 CfC 的历史聚合

对有效历史 mask `m_t`，计算：

```math
\bar x=\frac{\sum_t m_t x_t}{\max(\sum_t m_t,1)}
```

再取最后一个有效帧 `x_last`，由一个普通 MLP 融合：

```math
h=\tanh(\operatorname{LN}(W[x_{last},\bar x]+b))
```

这一步没有循环状态、时间常数、衰减门或 CfC 参数。`dt` 仍由数据入口校验，用于审计采样时间，但当前 GNN-only 网络不使用它。右侧 padding 严格被 mask 排除。

### 6.3 监督信号

GNN 与历史融合 MLP 通过多个任务头联合训练：

- phase / transition 事件分类；
- 当前 `held_by_actor`、`supported_by_target`、`in_target_region` 三个三分类关系头；
- 0.3 s 与 0.6 s 的未来关系头；
- 由显式目标关系条件化的 `not_satisfied / satisfied / unknown` 目标头。

这些任务头把梯度传回 GNN、ROI CNN、全局 CNN 和历史融合 MLP。发布结果时仍执行因果稳定确认：默认要求置信度至少 `0.6`，连续 3 次观察且跨度至少 `0.18 s`；时间断点超过 `0.151 s` 或输出 unknown 会清空确认计数。高目标分数不能覆盖已确认的关系矛盾。

evidence 版本另加当前视觉证据、pair visual 和遮挡 prior：有证据时直接用当前图表示校正，无证据时用最后一次可见运动驱动的 prior 延续 belief；其中同样没有 CfC observation update。

实现见 [`relational_observer.py`](../umi_cfc/_internal/event_rl/relational_observer.py) 与 [`evidence_observer.py`](../umi_cfc/_internal/event_rl/evidence_observer.py)。

## 7. 动作条件世界模型怎样预测关系后果

RoboTwin 分支把 VLA/机器人原生动作先转换成每一步 14-D 规范效果：

```text
left : Δx, Δy, Δz, Δroll_axis, Δpitch_axis, Δyaw_axis, Δgripper
right: Δx, Δy, Δz, Δroll_axis, Δpitch_axis, Δyaw_axis, Δgripper
```

这里的旋转实际用轴角增量表达；每步同时输入动作 `Δt`。一个 GRU 把整个候选 chunk 编码为 96-D action embedding。历史 typed graph 经三层 GNN 和 CfC 得到 `h_history`；语言由 byte-level GRU 编码，目标关系图也独立编码。四者融合：

```text
future = history + Transition(history, action, language, goal)
```

再预测：

- 当前和未来的 12 类节点对关系；
- 未来节点状态增量；
- 10 类事件（approach、grasp、transport、place、release、drop、regrasp、open、activate、idle）；
- 目标关系的组合满足概率；
- 成功 residual logit；
- 掉落概率。

训练损失联合包含当前/未来关系 BCE、未来节点 Smooth-L1、事件 BCE、语言目标重建、目标满足 BCE、成功 BCE、预测未来场景与真实未来场景的对比损失，以及小权重的 UMI legacy relation 迁移约束。

候选评分器目前使用：

```math
U=0.7P(\text{goal})+0.3P(\text{success})-0.35P(\text{drop})
```

这个公式是当前工程 scorer 的启发式 utility，不是经过策略回报校准的 Q 值。尤其当前 checkpoint 只有 10-step smoke，不能用于部署动作选择。

代码依据：[`train.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/train.py)、[`infer.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/infer.py)。

## 8. 从关系/事件到价值、优势和 VLA 调制

### 8.1 三个概念必须分开

1. `P(relation/event/goal | history)` 是观察器预测。
2. `V(s)` 或 `Q(s,a)` 是对未来累计回报的估计。
3. `A(s,a)` 是相对基线的优势，用于决定某个已执行动作应该被模仿得更多还是更少。

UMI GNN-only observer 只负责第 1 层，不能把“未来放置概率”直接命名为 value/advantage。历史主线 CfC-AWR 通过独立 value adapter、episode 级交叉拟合和 TD(`λ`) 产生优势；动作条件 RoboTwin 模型在结构上可支持第 2 层，但仍需正式数据、校准和 held-out 验证。

标准离线链路为：

```text
关系/事件序列
  -> 预先定义并审计的 event reward
  -> return / OOF value baseline
  -> TD(λ) advantage A_i
  -> w_i = clip(exp(A_i / β), w_min, w_max)
  -> VLA 原生损失加权
```

统一加权目标为：

```math
\mathcal L_{VLA}=\frac{\sum_i w_i\,\ell_i^{native}}{\sum_i w_i}
```

其中 `w_i≥0`，在进入策略损失前 `detach`，padding token/action 由各自 mask 排除。权重归一化避免仅因 batch 平均权重变化而改变整体梯度尺度。

### 8.2 为什么同一信号可以调制不同 VLA

因为跨 VLA 共享的是**样本级行为质量**，不是网络内部特征维度。对同一条 `(observation, instruction, executed action chunk)`，事件系统只回答“这段行为相对基线有多好”，输出一个标量 `w_i`。每个 VLA 仍以自己的输入、动作表示和训练目标计算 `ℓ_i^native`。

| VLA 类型 | 保持不变的原生损失 | 调制位置 |
|---|---|---|
| SmolVLA | `[B,T,D]` flow-matching loss | 先按真实 action 维和 token mask 归约成每样本 loss，再乘 `w_i` |
| OpenVLA 类 | action-token next-token cross entropy | 只保留 action token 和非 `-100` label，再按样本乘 `w_i` |
| pi0.5 类 flow actor | 模型自身未归约 flow loss | 由薄 adapter 保留 batch 维和 action mask，再乘 `w_i` |

这有四个好处：

- 不要求不同 VLA 共享 hidden size、tokenizer、action head 或 diffusion/flow 参数化；
- 不把 YOLO/GNN/CfC embedding 强行拼入 VLA token，避免修改推理 ABI；
- 不把策略梯度反传进 reward/value 计算，降低共同投机风险；
- 后训练产物仍是标准 VLA checkpoint，部署时可不加载 critic。

适配器代码已经提供 SmolVLA 0.4.4、HuggingFace action-token 和 callable flow 三类接口，见 [`adapters.py`](../experiments/cross_embodiment_event_rl_v1_20260903/event_rl/adapters.py) 与 [`awr.py`](../experiments/cross_embodiment_event_rl_v1_20260903/event_rl/awr.py)。但“有适配器”只证明接口可接，不证明 OpenVLA/pi0.5 已完成训练或性能验证。

### 8.3 哪些内容仍必须按 VLA/本体单独适配

下列部分不能假装完全通用：

- 相机命名、图像预处理、语言 tokenizer；
- VLA 原生 state/action schema、chunk 长度和 padding mask；
- 原生关节动作到 14-D EE 效果的运动学映射；
- 双臂/单臂缺失侧的规范表示；
- flow、diffusion、token CE 各自的未归约 loss 获取方式；
- checkpoint/processor 的完整性验证；
- 在线候选动作从策略空间到世界模型空间、再回到可执行空间的闭环延迟。

因此，系统的正确分层是“共享事件语义与标量价值接口，私有 VLA loss adapter 和本体 action adapter”，而不是宣称一个 checkpoint 可无条件插入所有 VLA。

## 9. 从仿真预训练、UMI 适配到真机微调部署

### 9.1 先区分三次训练和一次部署

完整链路包含三个不同的优化过程，更新的参数并不相同：

| 阶段 | 数据 | 被更新的参数 | 保持冻结 | 产物 |
|---|---|---|---|---|
| A. 仿真世界模型预训练 | RoboTwin 物理状态、关系、事件、14-D 动作和结果 | typed GNN、CfC、语言/目标/动作编码器、未来关系与风险头 | 可选的 UMI V4 初始化权重只是起点 | 动作条件事件世界模型 checkpoint |
| B. UMI 观察器与价值适配 | 真实视频、YOLO 图、审查关系/事件、真实终局 | UMI GNN-only observer；随后单独训练 value adapter | YOLOE；训练 value 时再冻结 observer | observer、OOF value、advantage/weight sidecar |
| C. 真机 VLA 后训练 | VLA 原生 RGB/state/action 与已绑定权重 | VLA actor 参数 | YOLOE、observer、value、sidecar | 新的标准 VLA `pretrained_model` |
| D. 真机部署 | 在线 RGB、state、任务文本 | 不训练，只推理 | 不加载训练期 critic | 动作 chunk |

因此“仿真预训练”“UMI 微调”和“VLA 微调”不是一次端到端反向传播。当前最稳妥的系统把世界模型当训练期教师，把策略当部署期学生。

### 9.2 阶段 A：RoboTwin 仿真世界模型怎样预训练

#### 9.2.1 采集和监督

采集器在每次成功物理步后按真实仿真时间记录一个 episode HDF5。核心字段为：

```text
node_features   [T,8,24]       typed node 的位姿、速度、状态
node_types      [T,8]
node_mask       [T,8]
edge_features   [T,8,8,12]     节点对相对几何和相对速度
relations       [T,12,8,8]     12 类关系监督
relation_mask   [T,12,8,8]     哪些关系有定义
events          [T,10]         grasp/place/drop/regrasp 等事件
goal/goal_mask  [12,8,8]       任务目标关系图
actions14       [T-1,14]       双臂 EE 位移、轴角和夹爪增量
sim_times       [T]            真实物理时间
success         [T]            仿真任务成功状态
```

物理接触、夹爪闭合、共同运动、功能点和任务 `check_success()` 用于构造监督标签，但机器人 ID、原生关节编号和直接接触位不进入共享模型。正常主动张开标记为 release；只有闭合夹爪下的非预期分离/下坠才标记为 drop，避免把成功放置误判为掉落。

采集入口是 [`collect_robotwin.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/collect_robotwin.py)，数据格式由 [`schema.py`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/universal_event/schema.py) 固定。采集完成后必须运行 `audit_dataset.py`，检查 schema SHA、时间单调性、关系/事件覆盖、动作有限性和成功/失败一致性。

#### 9.2.2 窗口怎样生成

默认训练样本由同一 episode 内的滑动窗口生成：

```text
过去 16 个图状态
+ 对应 history Δt
+ 当前任务语言和 goal graph
+ 从当前时刻开始的 30 步 actions14/action Δt
──────────────────────────────────────────────
监督第 30 步时的节点、关系、目标和成功
+ 监督这 30 步内是否出现各类事件
```

默认 `stride=3`。事件目标是未来动作窗口内逐类取最大值；未来节点监督只选择位置、关节开合、位移、高度、夹爪和速度等 8 个通道的变化。所有窗口保持在原 episode 内，不允许跨轨迹拼接。

episode 先按任务、本体、seed 和文件名哈希划分，默认 10% 作为 validation；指定的 held-out task/body 不进入 train/validation。需要注意：当前 `train.py` 只实例化 train 和 validation loader，held-out 数据虽然被排除，但还需要独立评估入口才能形成真正的 leave-task/body 结果，不能把“排除过”写成“已经测试过”。

#### 9.2.3 初始化、损失和优化

模型可以从 UMI Event V4 复制 12 个形状且语义兼容的 CfC/legacy relation tensors。它不是加载完整 UMI 网络；typed GNN、语言、动作和未来解码器仍从头学习。

联合目标为：

```math
\mathcal L=
0.5L_{rel,current}+2L_{rel,future}+0.5L_{node}
+L_{event}+0.5L_{language}+L_{goal}+L_{success}
+0.25L_{contrast}+0.1L_{legacy}
```

其中关系和事件使用带 mask 的 BCE，未来节点使用 Smooth-L1；对比项要求动作条件预测的未来场景靠近真实未来图编码。验证还会打乱 action chunk，记录目标概率变化；如果动作打乱几乎不影响输出，说明模型可能忽略动作，不能当作 action-conditioned world model 验收。

默认正式配置是 AdamW、学习率 `2e-4`、weight decay `1e-4`、batch 64、100,000 steps、cosine decay、梯度裁剪 2.0；每 1,000 步验证，每 5,000 步保存，按 validation 总损失选择 `best.pt`。建议运行三个随机种子。

```bash
python experiments/robotwin2_universal_event_pretrain_v1_20260909/train.py \
  --data /path/to/robotwin_universal_event_hdf5 \
  --output /new/output/seed_20260909 \
  --v4-checkpoint /path/to/event_v4_selected_observer.pt \
  --steps 100000 --batch-size 64 --history 16 --horizon 30 --stride 3 \
  --heldout-body piper \
  --heldout-tasks open_microwave place_empty_cup \
  --seed 20260909 --device cuda
```

输出目录必须不存在，训练会写入 protocol receipt、逐步 checkpoint、validation history 和 best checkpoint。当前仓库只有单任务数据上的 10-step smoke；三 seed 100k 命令是正式方案，不是已完成结果。

### 9.3 阶段 B1：当前可执行的 UMI YOLOE+GNN-only 观察器训练

UMI 路线首先训练“看懂真实视频发生了什么”的 observer，而不是直接训练机器人动作。

1. 准备按真实时间对齐的 UMI 视频帧、`attempt_uid/query_id/elapsed_s` 和 train/validation source-group split。
2. 用冻结 YOLOE 批量提取 object/target/gripper 检测；模型权重、prompt、阈值和图像 SHA 写入回执。
3. 用因果 tracker 生成最多 8 节点的二维角色图和 64×64 ROI；只向前解码视频，不允许未来插值。
4. 使用独立审查的关系/事件区间作为监督。空白标签按头 mask，明确审核的 unknown 作为真实类别训练。
5. 训练两层 GNN、全局/ROI CNN、历史融合 MLP、关系/事件/未来关系/目标头；YOLOE 不在 optimizer 中。

图缓存命令的输入必须与事件数据、检测 JSONL、视频 provenance 和原始视频一一绑定：

```bash
python umi_cfc/_internal/scripts/build_relational_graph_cache_v3.py \
  --data /path/to/origin_events.npz \
  --detections /path/to/origin_yolo_detections.jsonl \
  --provenance /path/to/source_videos.json \
  --raw /path/to/raw_videos \
  --output /path/to/origin_relational_graph_v3.npz
```

训练入口为：

```bash
python umi_cfc/_internal/scripts/train_relational_observer_v3.py \
  --data /path/to/origin_events.npz \
  --graph /path/to/origin_relational_graph_v3.npz \
  --annotations /path/to/relations_ai_reviewed.csv \
  --output /new/umi_relational_v3 \
  --steps 4000 --batch-size 16 --window 12 \
  --learning-rate 1e-4 --device cuda
```

V3 的实际联合损失是：

```math
0.5L_{relation}+0.25L_{event}+0.25L_{goal}
+0.2L_{forecast}+0.05L_{consistency}
```

每 200 步在 source-group 隔离的 validation 上评估，按关系、事件和目标的 episode-balanced CE 选择 checkpoint。V4 可在其上加入 pair visual、可观测性和遮挡 prior，但当前未通过晋级门，因此真机 actor 不应默认切换到 V4。

### 9.4 仿真 checkpoint 怎样迁移到 UMI：当前现实与目标方案

当前不能把 RoboTwin checkpoint 直接整模型加载到 UMI V3，原因是两个输入 ABI 不同：

```text
RoboTwin：3D typed nodes [8,24] + edges [8,8,12] + actions14
UMI V3 ：2D YOLO role nodes [8,15] + edges [8,8,7]，不含动作
```

目前代码实现的是 **UMI V4 → RoboTwin** 的 12 个兼容 tensor 初始化，而不是 **RoboTwin → UMI V3** 的完整 checkpoint 微调。因此，“先仿真预训练整个 GNN 世界模型，再直接在现有 UMI 二维图上 fine-tune”目前还不是一条真实可运行命令。

要打通统一的 sim-to-real UMI 微调，需要新增并验证以下适配层：

1. **真实 typed graph adapter**：利用标定后的多相机、深度/位姿估计或可靠跟踪，把真实物体、容器、支撑、关节物体和夹爪映射到与仿真一致的 24-D node/12-D edge schema。
2. **真实 canonical action adapter**：利用机器人 FK 和真实控制时间戳，把原生 `action25/state25` 转换为与仿真一致的双臂 14-D EE 增量；不能把 25-D 关节向量截断或补零冒充 actions14。
3. **真实监督适配**：把审核关系、掉落/重抓/释放和终局结果转换成相同 relation/event/goal mask；不可见事实保持 unknown/masked。
4. **分阶段解冻**：先冻结仿真 GNN/CfC，训练真实感知 adapter 和新输入投影；再以较小学习率解冻上层 GNN/CfC/heads，并混入仿真 replay 防止关系动力学遗忘。
5. **严格 sim-to-real 验收**：分别报告真实关系 F1、动作打乱敏感性、未来关系校准、留任务/留本体结果和闭环成功率。

这五项是推荐的统一架构实施方案，当前仓库尚未实现。现阶段可执行的是“仿真动作世界模型实验”和“UMI 二维关系观察器实验”两条并行路线，通过关系语义和部分 CfC 参数关联，而不是一个已经端到端打通的 sim-to-real checkpoint。

### 9.5 阶段 B2：从 UMI 观察器得到 value、advantage 和训练权重

当前正式真机 actor 使用的是已经验收的 original RGB-CfC/value 历史路线，不是新的 GNN-only observer。其步骤为：

1. 在 origin UMI 的 train/validation 上训练 RGB-CfC 事件观察器。
2. 冻结观察器，给 S1 LeRobot 真实轨迹提取 `96-D CfC history + 15-D event probabilities = 111-D` 因果视频特征。
3. S1 的 8 个 `state25` 历史点只进入独立 64-D state adapter，不回写共享视频表示。
4. 在 `[111-D video, 64-D state]` 上拟合标量 `V(s)`；按原始录像组做五折 OOF，使每个训练 episode 由没有见过该 episode 的 value fold 评分。
5. 用审核终局和事件回报计算连续时间折扣、TD(`λ`) advantage，并按真实 action chunk 边界生成 sidecar。

正式三任务协议使用约 10 Hz 观察、50 帧动作 chunk（约 1.67 s）、`λ=0.95`，权重为：

```math
w=\operatorname{clip}\left(\exp\left[\frac{A}{3\,\operatorname{scale}(A)}\right],0.9,1.1\right)
```

unknown 终局保持 `w=1`，不当成失败。sidecar 必须用全局 `index/episode_index/frame_index` 与 actor 数据逐行绑定，并检查 chunk endpoint 不跨 episode。

```bash
# 训练 origin RGB-CfC
python umi_cfc/_internal/cfc_value_umi_s1_20260906/pipeline.py pretrain \
  --umi /path/to/origin_cache.npz --output /new/pretrain \
  --steps 2000 --device cuda

# 冻结观察器，提取真实 S1 因果特征
python umi_cfc/_internal/cfc_value_umi_s1_20260906/pipeline.py extract \
  --data /path/to/s1_lerobot_v3 \
  --checkpoint /path/to/selected_observer.pt \
  --output /path/to/s1_features.npz --device cuda

# 五折 value、TD(lambda) 和 AWR sidecar
python umi_cfc/_internal/cfc_value_umi_s1_20260906/value.py \
  --features /path/to/s1_features.npz \
  --labels /path/to/terminal_labels.json \
  --output /new/critic --steps 1500 --device cuda
```

如果未来让 GNN-only observer 取代 original RGB-CfC，必须重新完成训练、独立标注验证、OOF value 拟合、权重排序/ESS 审计和同预算 actor 对照；不能直接把关系概率当 advantage。当前发布的三个真机 SmolVLA checkpoint 并不是由新 GNN-only observer 评分得到的。

### 9.6 阶段 C：怎样在真机数据上后训练 VLA

以当前正式三相机 SmolVLA 为例，策略后训练需要四项严格绑定的输入：

```text
纯 SmolVLA 050000/pretrained_model 及 model SHA
三相机 LeRobot v3 数据及连续全局 index
与每个 index 精确对齐的 event_weights.parquet
固定任务 prompt、state25/action25、chunk_size=50 契约
```

三路图像键为：

```text
observation.images.base_0_rgb
observation.images.left_wrist_0_rgb
observation.images.right_wrist_0_rgb
```

preflight 会检查源 checkpoint SHA、observer/sidecar SHA、相机集合、任务文本、episode/frame 数、state/action 维数、chunk endpoint，以及 actor/reference 数据值是否一致。任一项不匹配即停止，而不是尝试自动修正。

训练时 SmolVLA 仍计算自己的 `[B,T,D]` flow-matching loss；只保留真实 25-D action 和有效 token，先归约为 per-sample loss，再乘已 detach 的事件权重。YOLOE、observer 和 value 均不进入 actor optimizer。水果/锅盖正式配置为额外 10,000 optimizer steps、学习率 `1e-6`、batch 32、100-step warmup、cosine decay、每 1,000 步保存；应先跑 5-step smoke，再提交正式作业。

```bash
cd real_robot/cfc_awr_three_camera

# 机械接线检查
sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=5 \
  train_three_camera_cfc_awr.slurm

# 新目录中的正式后训练
sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=10000 \
  train_three_camera_cfc_awr.slurm
```

训练必须从已审计的纯 50k checkpoint 开始，写到全新的输出目录，不能覆盖源模型。训练器保持源 processor/normalizer 文件字节级一致；验收要求：保存点完整、最终 actor 权重确实变化、相机/state/action/chunk ABI 不变、processor SHA 不变、最终目录能被标准 SmolVLA loader 重载。

蔬菜任务使用 `umi_cfc/_internal/cfc_value_umi_s1_20260906/train_hpc.slurm`；水果和锅盖使用 [`real_robot/cfc_awr_three_camera/train_three_camera_cfc_awr.slurm`](../real_robot/cfc_awr_three_camera/train_three_camera_cfc_awr.slurm)。不同任务必须重新绑定各自数据、prompt、sidecar 和 source SHA，不能复用另一任务的权重文件。

### 9.7 阶段 D：怎样放到真机部署

验收通过后，部署的在线数据流是：

```text
base RGB + left wrist RGB + right wrist RGB + state25 + task prompt
                              │
                              ▼
                原 checkpoint 配套 processor/normalizer
                              │
                              ▼
                  后训练后的 SmolVLA checkpoint
                              │
                              ▼
                       50 × 25 动作 chunk
                              │
                              ▼
                 现有机器人控制器与安全限制
```

部署包是最终 `checkpoints/010000/pretrained_model` 完整目录，而不是单独复制 `model.safetensors`。配置、processor JSON、normalizer/unnormalizer 权重和模型必须来自同一验收目录。

部署时不需要：

- YOLOE 权重或 prompt；
- UMI 图缓存；
- GNN/CfC observer；
- value fold 或 AWR sidecar；
- RoboTwin 仿真世界模型。

上线前仍需在目标机器做只读加载、三相机键/尺寸检查、state/action 维度检查、静态 batch 推理和动作有限性/范围检查，再按现有机器人安全流程做低速、限幅、急停可用的分阶段真机测试。必须保留纯 50k checkpoint 作为可回退基线，并用相同任务种子/初始条件与后训练模型做成对评估。

如果未来采用“在线候选重排”，才需要在部署时额外运行真实 typed graph adapter、14-D action adapter 和 GNN+CfC scorer；这条路线当前未完成，不属于现有三任务部署契约。

### 9.8 当前实际打通程度

```text
RoboTwin 仿真采集/模型代码 ── 已有
RoboTwin 10-step smoke      ── 已完成
RoboTwin 多任务 100k 预训练 ── 未完成
仿真 checkpoint → 统一 UMI 3D 图微调 ── 未实现
旧 UMI 二维 GNN+CfC V3 observer ── 已训练，仅作历史对照
新 UMI YOLOE+GNN-only observer ── 代码已完成，尚未重新训练
GNN-only observer → 正式 OOF value → actor ── 未完成训练与正式验证
original UMI RGB-CfC → OOF value → SmolVLA AWR ── 已完成
后训练 SmolVLA 单独真机部署契约 ── 已完成
```

这意味着当前可部署资产证明的是“UMI RGB-CfC/value 辅助的离线 SmolVLA 后训练可以产出标准策略 checkpoint”，而不是“RoboTwin GNN+CfC 已经完成 sim-to-real 并驱动真机策略”。

## 10. 当前实现与证据状态

| 部分 | 当前状态 | 已有证据 | 不能宣称 |
|---|---|---|---|
| 冻结 YOLOE 线索 | 已实现 | 本地权重、prompt、34-D cue 和图 cache 构建代码 | YOLO 自动产生关系/事件真值 |
| UMI YOLOE+GNN-only | 代码已修改，待重新训练 | CfC 参数与更新已移除；核心单测通过 | 沿用旧 V3/V4 指标或 checkpoint |
| 旧 UMI GNN+CfC V3 | 历史对照 | 106 段 origin，85 train / 21 validation；4,000-step 预算，选 step 2,200；内部 validation CE `0.4711`，no-graph `0.5023` | 新 GNN-only 模型效果 |
| UMI V4 evidence | 已训练但未晋级 | 选 step 2,600；失败点假成功下降，但 unknown `73.40%`，盘外仍有 11 个 false-confirmed-success 点 | 可靠奖励或默认 observer |
| RoboTwin 采集 | 小规模完成 | 24 条 clean HDF5；任务 roster 12、本体 roster 5；22 pass、2 个保留的真实专家失败 | 12×5 完整覆盖或大规模多本体数据集 |
| RoboTwin GNN+CfC | 只完成 smoke | 单任务数据 532 train / 68 validation、10 optimizer steps，验证前向/反向/保存和 12 个兼容张量迁移 | 正式预训练、动作排序效果、零样本跨本体 |
| 跨 VLA loss adapter | 代码接口完成 | SmolVLA flow、OpenVLA 类 token CE、pi0.5 类 callable flow contract | 三种 VLA 均已训练或均有收益 |
| Event-AWR 策略交付 | SmolVLA 路线已完成既有后训练 | 三任务 SmolVLA checkpoint/回执；critic 仅训练期使用 | GNN+CfC 世界模型已驱动所有已发布策略 |

UMI V3 在 1,200 个已标目标失败点上有 3 次假成功，no-graph 为 5 次；但 V3 的成功召回 `68.21%`，低于 no-graph 的 `75.50%`。这说明图结构降低了一部分误报，但不能只报一个方向的改进。以上均是内部 AI 标注验证，不是人工金标准独立测试。

详细分支状态见 [`GNN_CFC_BRANCH.md`](GNN_CFC_BRANCH.md)；仿真实验状态见 [`RoboTwin README`](../experiments/robotwin2_universal_event_pretrain_v1_20260909/README.md)。

## 11. 建议的完整落地顺序

要把当前原型变成真正可跨 VLA 使用的完整框架，应按以下顺序收口：

1. **冻结输入契约**：明确每个任务的角色、关系、goal graph、相机和 14-D action adapter；禁止 body id 泄漏进共享模型。
2. **扩充并审计数据**：完成计划中的多任务×多本体×成功/失败采集；保留掉落、误抓、重抓、遮挡和恢复，而不只收专家成功。
3. **分层验证感知**：先独立测 YOLO 角色检测、track/binding 和关系标注质量，再训练世界模型，避免把检测错误隐藏在端到端 loss 中。
4. **正式训练世界模型**：episode 级划分，至少做 leave-one-body-out 和 leave-one-task-out；检查 action shuffle sensitivity，证明模型确实使用动作。
5. **校准价值接口**：把未来关系/事件转换为固定 reward，使用 OOF value/TD(`λ`) 或执行结果校准；单独报告成功、掉落和恢复子集。
6. **接入 VLA 原生 loss**：每个 VLA 只实现未归约 loss adapter；保持 tokenizer、processor、action schema 和初始 checkpoint 可审计。
7. **先离线后训练，再做在线重排**：离线 AWR 风险更可控；在线 scorer 必须额外通过延迟、动作转换、候选多样性和真机安全验证。
8. **做成对评估**：同一初始化、同一数据、同一训练步数比较纯 BC 与 Event-AWR；分别报告各本体/任务和失败类型，不只给混合平均数。

## 12. 代码与资产定位

| 内容 | 路径 |
|---|---|
| YOLOE 冻结检测与 34-D cues | `umi_cfc/_internal/event_rl/semantic_cues.py` |
| UMI 因果角色图 | `umi_cfc/_internal/event_rl/relational_graph.py` |
| UMI 两层 GNN-only observer | `umi_cfc/_internal/event_rl/relational_observer.py` |
| V4 evidence / prior | `umi_cfc/_internal/event_rl/evidence_observer.py` |
| 当前 YOLOE 权重 | `umi_cfc/weights/yoloe-11s-seg.pt` |
| 当前角色 prompt | `umi_cfc/weights/yolo_prompts.json` |
| V3 active model 指针 | `umi_cfc/weights/active_model.json` |
| RoboTwin schema | `experiments/robotwin2_universal_event_pretrain_v1_20260909/universal_event/schema.py` |
| RoboTwin typed GNN/CfC/model heads | `experiments/robotwin2_universal_event_pretrain_v1_20260909/universal_event/model.py` |
| 仿真采集和规范动作 | `experiments/robotwin2_universal_event_pretrain_v1_20260909/collect_robotwin.py` |
| 仿真数据审计 | `experiments/robotwin2_universal_event_pretrain_v1_20260909/audit_dataset.py` |
| 多任务训练 | `experiments/robotwin2_universal_event_pretrain_v1_20260909/train.py` |
| 候选 chunk scorer | `experiments/robotwin2_universal_event_pretrain_v1_20260909/infer.py` |
| UMI 图缓存构建 | `umi_cfc/_internal/scripts/build_relational_graph_cache_v3.py` |
| UMI GNN-only 训练 | `umi_cfc/_internal/scripts/train_relational_observer_v3.py` |
| 正式 original CfC 预训练/特征提取 | `umi_cfc/_internal/cfc_value_umi_s1_20260906/pipeline.py` |
| 五折 value/TD(`λ`)/sidecar | `umi_cfc/_internal/cfc_value_umi_s1_20260906/value.py` |
| 跨 VLA loss adapter | `experiments/cross_embodiment_event_rl_v1_20260903/event_rl/adapters.py` |
| AWR 权重与归约 | `experiments/cross_embodiment_event_rl_v1_20260903/event_rl/awr.py` |
| 三相机 SmolVLA 后训练 | `real_robot/cfc_awr_three_camera/train_three_camera_cfc_awr.slurm` |
| Actor preflight | `real_robot/cfc_awr_three_camera/code/preflight_three_camera_cfc_awr.py` |
| Actor 输出验收 | `real_robot/cfc_awr_three_camera/code/verify_three_camera_actor.py` |

## 13. 一句话对外表述

可以准确表述为：

> 我们构建了一个实验性的 YOLOE+GNN 关系学习框架：冻结 YOLOE 为真实视频提供可审计的角色候选和 ROI，GNN 学习物体、目标与夹爪之间的关系，当前图与历史图表示通过无递归的掩码池化融合，再预测关系、事件和目标满足。新观察器不含 CfC，尚需重新训练和独立验证。

不应表述为：YOLOE 自动生成了可靠奖励、GNN+CfC 已完成大规模多本体预训练、或者同一 checkpoint 已在不同 VLA 上验证有效。
