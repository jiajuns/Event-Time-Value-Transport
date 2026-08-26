# ETSF × OpenVLA-OFT：跨本体 Critic 迁移完整技术路线

## 0. 文档定位

本文给出 Event-Time Value Transport（ETSF）的完整主实验路线。最终目标不是单独训练一个事件价值网络，而是：

> 以 RoboTwin 上的 OpenVLA-OFT 为策略 baseline，在源机械臂上训练共享的失败感知 critic；换到未见机械臂时冻结 OpenVLA 与共享 critic，只用少量目标 rollout 估计事件可达性和本体时钟，再验证 critic 是否能够改善候选动作排序、在线成功率和目标样本效率。

状态标记：

- **已完成**：已有代码、数据和正式结果。
- **开发性完成**：已在反复查看过的 Piper/UR5 上验证，不能作为确认性结论。
- **待实现**：尚无代码或尚未运行。

必须明确：当前 `run_stage3.py` 是对象状态事件价值的机制实验，**没有加载 OpenVLA-OFT**。它证明了时钟迁移和失败状态建模具有可行性，但不是最终 OpenVLA baseline 实验。

### 0.1 4090 对接审计结论

本文中的“4090”指已记录的远端 RTX 4090 D、24 GB 环境，不是文档所在机器的自动探测结果。当前仓库也没有 RLinf/OpenVLA-OFT 源码或 checkpoint，因此本次对接先冻结接口、资源边界和验收门；只有把 RLinf 与 RoboTwin 的实际路径、commit 和 checkpoint 补进运行 manifest 后，才能称为代码级接通。

已核实的官方接口基线是：

- OpenVLA-OFT 官方给出的显存需求约为 LIBERO 推理 16 GB、ALOHA 推理 18 GB；默认 BF16 训练配置需要约 27–80 GB。单张 24 GB 4090 适合做冻结策略推理、特征抽取和小型 critic 训练，不应承诺默认配置下的 OpenVLA 7B 全量训练。
- RLinf 的 RoboTwin OpenVLA-OFT 示例使用 `implement_version: official`、`action_dim: 14`、`proprio_dim: 14`、`num_action_chunks: 25`、任务相关 `unnorm_key`，并以 `ROBOT_PLATFORM=ALOHA` 为公开评测示例。
- 上述 `14` 维接口是 ALOHA 双臂接口，**不能在未验证 wrapper 和动作语义前直接套到 Piper 或 UR5-WSG**。
- RLinf 公开示例能够证明 RoboTwin 与 OpenVLA-OFT 已有 ALOHA 接口，但不能证明本文六任务、四本体和所有任务 checkpoint 已经齐全。

因此新增三个前置硬门：

```text
A0 action_contract：每个本体的 action/proprio 维度、单位、顺序、控制模式和 unnorm_key 全部通过回放测试
A1 event_contract：线上 critic 不读取仿真器真值 event token；真值事件只作训练标签和 oracle 消融
A2 proposal_contract：明确候选 action chunk 如何产生；默认确定性 L1 动作头不能被描述为“采样 N 次”
```

三门任一未通过，只能运行 B0 和离线特征抽取，不能报告跨本体 critic 对接完成。

---

## 1. 研究问题

### 1.1 实际问题

OpenVLA-OFT 在源机械臂上训练后，策略和 critic 会同时受到以下本体差异影响：

- 关节结构和动作空间不同；
- 控制器、控制频率和动作 chunk 实际持续时间不同；
- 相同语义子任务在不同机械臂上的可达率不同；
- 失败形态不同，例如抓空、夹持不稳、动作范围不足和事件链中断；
- 图像外观、相机位置和机器人自身遮挡不同。

若在每个目标本体上重新训练 critic，需要重新采集奖励、执行目标域 TD，并承担长视界信用分配成本。本项目要检验：能否把目标端学习量限制为低维时钟和事件后验，同时保留源 critic 的语义知识。

### 1.2 三个可证伪假设

**H1：语义—时钟可分离。**

任务进度、失败状态和事件后继语义可以跨本体共享；本体主要通过事件持续时间分布和事件可达率影响最终 discounted value。

**H2：失败监督能够解除事件计数退化。**

只使用成功 demonstration 时，事件 successor 头容易退化为 `V=f(task,event_index)`。加入匹配任务、事件和初始条件的失败状态后，完整状态模型应在同事件 success/failure 排序上超过事件常数基线 AUC=`0.5`。

**H3：事件价值能够帮助动作 critic。**

只有当 ETSF 状态价值接入动作条件 critic 后，能够改善候选动作排序和 OpenVLA 闭环成功率，才可以称为 critic 迁移；独立事件头只能称为 event-value transport。

---

## 2. 最终系统结构

```text
RGB 图像 + 语言指令 + proprio + 动作历史
                    │
                    ▼
             OpenVLA-OFT 骨干
                    │
             多模态状态特征 h_t
          ┌─────────┴──────────┐
          │                    │
          ▼                    ▼
   OpenVLA-OFT 动作头      critic 特征桥 Pθ
                               │
                     对象中心/事件表示 z_t
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             语义 successor 头       ClockLNN 时钟头
             Ψθ(z_t)，β 不可见       qθ(D|z_t,β_b)
                    │                     │
                    └──────────┬──────────┘
                               ▼
                 EventTransport(Ψ,E[γ^D],ρ_b)
                               │
                         V_ETSF(s_t)
                               │
              ┌────────────────┴───────────────┐
              ▼                                ▼
       动作优势头 A(s,a)                候选动作后果/价值头
              │                                │
              └────────────────┬───────────────┘
                               ▼
                Q(s,a)=V_ETSF(s)+A(s,a)-E[A]
                               │
                               ▼
                     OpenVLA 候选动作排序
```

### 2.1 OpenVLA-OFT baseline

OpenVLA-OFT 是最终的策略 baseline，负责：

- 图像和语言编码；
- proprio/action token 处理；
- 动作 chunk 生成；
- 提供策略隐藏特征；
- 产生真实 on-policy 成功与失败 rollout。

必须先在 RoboTwin 六个任务上复现 OpenVLA-OFT 原始成功率，冻结 checkpoint、预处理、动作反归一化、chunk 长度和控制频率。未完成复现前，不运行 ETSF 主对比。

### 2.2 critic 特征桥

特征桥把 OpenVLA 隐状态映射到 critic 空间：

```text
z_t = Pθ(h_t, task_token, soft_event_posterior, object_state_optional)
```

第一版建议同时保留两种输入：

- OpenVLA 隐状态：视觉、语言、机器人状态和动作上下文；
- 仿真器对象状态：仅在开发阶段作为 oracle feature baseline。
- 软事件后验：部署时由 `EventPosterior(h_t,task,history)` 预测，不能读取仿真器真值事件。

由此形成：

- `OpenVLA-only critic`：真实部署接口；
- `object-state oracle critic`：判断失败是否来自表征不足；
- `OpenVLA + object state`：开发上界，不作为最终部署方案。

### 2.3 共享语义 successor 头

语义头输出事件后继可达性或规范化价值分布：

```text
Ψθ(z_t) ∈ R^(K×B)
```

约束：

- β、本体 ID和原始速度不进入语义头；
- HL-Gauss support 只由源验证集冻结；
- 时钟损失通过 `stop-gradient(z)` 进入 ClockLNN，不能重写语义空间；
- 同事件存在多个观测时，使用最后观测评估状态质量；
- 失败终止状态沿用最后已达事件 token，不输入显式 failure token。

### 2.4 ClockLNN

ClockLNN 只建模事件持续时间分布：

```text
log τ_b(z) = aθ(z) + c·tanh(β_b)·gθ(z)
qθ(log(1+D_j) | z_j, β_b)
```

输出必须是分布而非单点，以便：

- 推断 β 后验；
- 表达不同事件段的不确定性；
- 计算 `E[γ^D]`；
- 在观测不足时触发回退。

ClockLNN 的必要性必须通过相同参数量的 duration MLP、GRU-time 和标准 CfC 消融验证。当前结果只证明“可适配时钟分支有效”，尚未证明 LNN 是唯一选择。

### 2.5 显式半马尔可夫传输

最终事件价值不由 β 直接修改 logits，而由显式传输算子产生：

```text
V_{b,j} = ρ_{b,j} [r_j + E_{β,D}(γ^D) V_{b,j+1}]
```

当前实现使用 Gauss–Hermite 积分计算：

```text
E[γ^D]，而不是错误的 γ^E[D]
```

必须区分最终算子与现有代码：当前 `run_stage3.py::transport_values` 实现的是“语义事件值 × 可达门 × 事件链累计折扣”，尚未实现上式带事件奖励和动作 bootstrap 的完整递归 Bellman critic。上式是 OpenVLA 动作 critic 阶段的目标接口，不是当前 Stage 3 已完成项。

### 2.6 动作条件 critic

最终 critic 使用 dueling 分解：

```text
Q(s,a)=V_ETSF(s;β,ρ)+A_native(s,a)-E_{a'~π}[A_native(s,a')]
```

动作优势头必须接收：

- OpenVLA 状态特征；
- 候选 action chunk；
- 必要时接收 action-conditioned world-model future feature。

在这一阶段之前，不把事件边界 TD 增量称为策略 advantage。

### 2.7 与 RLinf/OpenVLA-OFT 的模块契约

建议把对接拆成五个窄接口，避免直接修改 OpenVLA 主干：

```text
RoboTwinEnv
    │ obs / proprio / instruction / body_id
    ▼
OpenVLAOFTPolicyAdapter ──► ActionContract ──► env.step(native_action)
    │
    ├── pooled_state_feature h_t ──► RolloutWriter
    │                                  │
    │                                  ▼
    │                            offline ETSF trainer
    │
    └── nominal action chunk + candidate chunks
                                       │
                                       ▼
                              ETSFActionCritic.rank()
```

接口职责：

| 模块 | 输入 | 输出 | 约束 |
|---|---|---|---|
| `OpenVLAOFTPolicyAdapter` | RLinf observation | 归一化 chunk、反归一化 chunk、池化状态特征 | 冻结模型；记录 checkpoint/processor/unnorm_key 哈希 |
| `ActionContract` | 规范动作 chunk、本体 ID | 原生控制动作和执行 mask | 显式记录维度、单位、绝对/增量、关节/笛卡尔、夹爪编码和控制频率 |
| `EventPosterior` | OpenVLA 特征、任务 token、历史 | 软事件后验 `p(e_j|h_t)` | 部署时不得读取对象真值；对象状态事件只监督该头 |
| `RolloutWriter` | 环境、策略和事件记录 | 可重放 episode | 图像可单独压缩；逐步保存特征、动作、时间戳、错误类型和哈希 |
| `ETSFActionCritic` | 状态特征、软事件后验、候选 chunk | `V_ETSF`、`A`、`Q`、不确定性 | 不反传到冻结 OpenVLA；候选集必须包含原始策略 chunk |

统一策略步输出最少应包含：

```text
PolicyStep = {
  state_feature:       float16[B,d],
  action_chunk_norm:   float32[B,H,A],
  action_chunk_env:    float32[B,H,A_native],
  executed_mask:       bool[B,H],
  action_contract_id:  string,
  checkpoint_hash:     string,
  processor_hash:      string,
  unnorm_key:          string
}
```

`state_feature` 默认取最后一层与动作查询相关的 token 后做定长池化，不保存整个 7B 主干的所有层 hidden state。具体 token 索引必须针对所用 RLinf commit 做一次 shape probe 并写入 manifest，不能凭模型名称硬编码。

### 2.8 两种跨本体实验协议不能混用

**协议 P-critic（推荐的首个正式实验）**：每个本体使用已冻结、已复现的本体原生 OpenVLA-OFT policy/action adapter；只检验共享 critic 是否迁移。它回答“critic 能否跨本体迁移”，但不声称同一个 actor 零样本控制新机械臂。

**协议 P-full**：源本体 actor 原样冻结，通过预先定义的统一笛卡尔动作空间和本体控制适配器直接控制未见机械臂。它同时检验 actor 与 critic 跨本体迁移，难度更高；若目标本体需要重新训练动作头，就不再是该协议。

主表先采用 P-critic。P-full 单列为扩展实验，否则 actor 失效会把 critic 迁移结论一起污染。

### 2.9 候选动作生成与 4090 推理方式

OpenVLA-OFT 的默认连续 L1 动作头输出确定性 action chunk，不能仅重复前向就得到独立候选。B7/B8 采用以下预注册方案之一：

1. `nominal + bounded perturbation`：保留原始 chunk，再在归一化动作空间施加有界、时间平滑扰动；第一版推荐。
2. `stochastic proposal head`：另训轻量条件分布，只负责提出候选，不改变 OpenVLA 主干。
3. diffusion 候选：仅作为额外 baseline；它已改变 OFT 的默认动作解码配方，不能与 L1-OFT 混称同一 baseline。

单卡 4090 按三种进程模式运行：

| 模式 | 4090 中常驻模块 | 目的 |
|---|---|---|
| 特征采集 | BF16 冻结 OpenVLA-OFT＋单环境＋轻量池化 | 一次策略前向同时写动作和定长特征 |
| critic 训练 | 仅 critic 和预提取特征 | 不加载 7B actor，给 batch 和消融留显存 |
| 在线排序 | BF16 冻结 actor＋轻量 critic | 每个控制步只跑一次 actor；N 个候选只重复轻量 critic 前向 |

若单环境仿真与 18 GB 左右的 ALOHA actor 同卡仍超显存，优先把环境与策略拆进程/拆设备或降低渲染并发；量化 actor 必须单列为 baseline 变体，不能悄悄改变 B0 checkpoint 口径。

---

## 3. 数据路线

### 3.1 当前已有数据

六个任务：

- `adjust_bottle`
- `handover_block`
- `move_can_pot`
- `place_container_plate`
- `beat_block_hammer`
- `lift_pot`

四个本体：

- Aloha、ARX-X5：600 条筛选成功 demonstration；
- Piper、UR5-WSG：每任务每本体 20 条未筛选 rollout，共 240 条，包含完整物体状态和成败标签。

当前目标 rollout 由 RoboTwin `task.play_once()` 脚本专家生成，**不是 OpenVLA rollout**。这些数据只能用于机制开发。

### 3.2 当前三本体留一开发协议

```text
折 1：Clock={Aloha,ARX}，失败语义来源=UR5，完全留出=Piper
折 2：Clock={Aloha,ARX}，失败语义来源=Piper，完全留出=UR5
```

作用：

- 零成本验证失败监督能否跨本体迁移；
- 给状态相关 `gθ(z)` 提供第三个真实本体；
- 保持每折被留出本体不参与共享训练。

限制：Piper、UR5 已反复用于开发，不能再提供确认性结论。

### 3.3 必须新增的 OpenVLA rollout

每个任务、本体、初始 seed，应由 OpenVLA-OFT 重复执行多次：

```text
M 个相同初始条件 × 不同策略采样噪声
```

每条 rollout 保存：

- 全相机 RGB；
- 语言指令；
- proprio；
- OpenVLA 输入与隐藏特征；
- 反归一化动作 chunk和实际执行动作；
- 对象状态；
- 事件时间戳；
- success/failure；
- failure onset 或最后可靠状态；
- timeout、环境初始化错误和执行失败分离标签。

相同初始状态的重复 rollout 用于估计经验成功率，避免用一次随机结果错误评估 `V(s0)`。

### 3.4 失败轨迹处理

失败轨迹不是坏 demonstration，而是有效后果监督：

| 模块 | 成功轨迹 | 失败轨迹 |
|---|---|---|
| OpenVLA 动作模仿 | 使用 | 屏蔽 |
| 语义 successor / MC value | 使用 | 使用 |
| 成功概率和排序 | 使用 | 使用 |
| ρ 事件后验 | 使用 | 使用 |
| ClockLNN 主实验 | 使用干净转移 | 默认不使用失败 episode |
| ClockLNN 生存分析消融 | 使用 | 已完成段使用，未完成段右删失 |
| future/world prediction | 使用 | 使用 |

失败终止边界：

- 使用失败 rollout 最后一帧或预注册 failure-onset 帧；
- 沿用最后已达事件 token；
- 不向编码器输入 failure 标志；
- 与成功轨迹同任务、同事件、匹配 seed 的状态构成排序对。

### 3.5 RoboTwin randomized 数据边界

RoboTwin `demo_randomized` 主要随机化场景和视觉条件，官方采集流程会搜索可成功的 seed后保存 demonstration。它不能在未审计前被当作 failure-rich 数据。

下载前先检查：

- 是否保存失败 episode；
- 是否有逐帧状态和动作；
- 是否区分 simulator error 与 policy failure；
- 是否包含所需机械臂；
- 是否与当前 RoboTwin 版本和任务定义一致。

---

## 4. 数据划分与防泄漏

### 4.1 三层数据

**开发集**：当前 Piper、UR5 数据，用于结构选择和错误定位。

**预注册验证集**：新采但允许调参的数据，只用于冻结 loss 权重、β 先验和门槛。

**确认集**：全新本体或密封 fresh rollout；模型、种子、指标和排除规则冻结后只运行一次。

### 4.2 目标适配

保持当前协议：

- 每任务前 5 条用于 β/ρ 适配；
- 每个目标本体实际使用 30 条适配 rollout；
- 适配时禁止目标 TD、目标 return、测试标签和 early stopping；
- 测试集与适配集按索引和 SHA-256 冻结。

### 4.3 禁止的泄漏

- 不使用测试成功标签选择 β；
- 不用 `eK` 末边界 AUC作为主指标；
- 不把不同任务的正负轨迹混成 pooled AUC；
- 不把同一测试集上的多个训练种子当作独立数据样本；
- 不把环境初始化错误当作策略失败；
- 不用被留出本体训练 OpenVLA、语义头或动作 critic。

---

## 5. 训练目标

### 5.1 OpenVLA baseline

```text
L_policy = L_OpenVLA_OFT
```

先复现官方或 checkpoint 对应的训练/评测配置，不与 ETSF 联合优化。

### 5.2 语义价值训练

```text
L_sem = L_HL-Gauss-successor
      + w_mc L_event-MC-return
      + w_rank L_matched-task-event-ranking
      + w_cal L_success-calibration
```

排序对要求：

- 同任务；
- 同事件；
- 优先匹配相同初始 seed；
- 使用该事件的最后无泄漏 prefix；
- `eK` 不参与 failure branch AUC。

### 5.3 时钟训练

```text
L_clock = -log qθ(log(1+D)|z,β)
```

主实验只使用干净、完整、规范下一事件已到达的转移，并保持：

```text
duration >= 5
```

### 5.4 动作 critic

```text
L_Q = L_distributional_TD
    + w_pair L_candidate-action-ranking
    + w_cons L_V/Q-consistency
```

若多个损失共享 OpenVLA 特征，先记录梯度范数和余弦；仅在确认长期梯度冲突后比较 PCGrad/CAGrad。

### 5.5 总损失和梯度隔离

```text
L_total = L_policy
        + w_sem L_sem
        + w_clock L_clock
        + w_Q L_Q
```

约束：

- 第一阶段冻结 OpenVLA，只训练 critic bridge/head；
- ClockLNN 接收 `stop-gradient(z_sem)`；
- 失败轨迹屏蔽 `L_policy`，保留 `L_sem/L_Q/future`；
- 目标本体适配只更新 β/ρ 后验。

---

## 6. Baseline 与消融矩阵

| 编号 | 模型 | OpenVLA | 失败监督 | ClockLNN | 动作条件 | 目的 |
|---|---|---|---|---|---|---|
| B0 | OpenVLA-OFT | 是 | 否 | 否 | 原动作头 | 原始策略 baseline |
| B1 | OpenVLA＋MLP critic | 是 | 成败 | 否 | 是 | 普通 critic baseline |
| B2 | OpenVLA＋事件查找表 | 是 | 否 | 否 | 否 | 塌缩下界 |
| B3 | OpenVLA＋语义 critic | 是 | 成败 | 否 | 否 | 失败状态价值 |
| B4 | OpenVLA＋语义 critic＋duration MLP | 是 | 成败 | MLP | 否 | 非 LNN 时钟对照 |
| B5 | OpenVLA＋语义 critic＋GRU-time | 是 | 成败 | GRU | 否 | 序列时钟对照 |
| B6 | OpenVLA＋语义 critic＋ClockLNN | 是 | 成败 | 是 | 否 | ETSF 状态价值 |
| B7 | B6＋动作优势头 | 是 | 成败 | 是 | 是 | 完整 ETSF critic |
| B8 | B7＋候选后果模型 | 是 | 成败 | 是 | 是 | inference-time ranking |
| B9 | target-from-scratch critic | 是 | 目标 TD | 任意 | 是 | 目标训练样本上界 |

所有架构比较使用：

- 配对初始化；
- 相同数据顺序；
- 相同训练预算；
- 相同 OpenVLA checkpoint；
- 至少 5 个开发种子；
- 确认阶段按功效分析冻结种子和 rollout 数。

---

## 7. 评估指标

### 7.1 时钟机制

- duration NLL；
- duration MAE；
- 事件快慢方向准确率；
- β 后验均值、宽度、覆盖和回退率；
- event MC return MSE；
- `E[γ^D]` 校准。

### 7.2 事件语义和失败分支

- 同任务、同事件 success/failure AUC；
- 相对事件常数基线 `0.5` 的增益；
- 匹配 pair 数和独立失败 prefix 数；
- failure detection time；
- event-only 与 full-state 的条件 NLL/Brier 差；
- failure-terminal 消融；
- 事件 token 移除/打乱 probe。

注意：成功轨迹内部进度准确率的事件索引基线已经是 `1.0`，不能要求模型超过它，也不能用 `0.99<1.0` 单独判定塌缩。

### 7.3 初始状态价值

同一初始 seed 重复执行 M 次，计算经验成功概率：

```text
p_hat(success|s0)
```

使用：

- Brier score；
- log loss；
- reliability diagram；
- calibration error。

单条 rollout 的成败 AUC只作诊断。

### 7.4 动作 critic

- 同状态候选动作 pairwise accuracy；
- `Q(s,a)` 与真实候选 rollout return 的相关性；
- top-1/top-k candidate selection regret；
- advantage calibration；
- OOD action false-positive rate。

### 7.5 在线策略

- OpenVLA 原始成功率；
- ETSF critic-guided 成功率；
- 平均 return；
- failure recovery rate；
- 达到同成功率所需目标数据量；
- 推理延迟和候选动作数量；
- 每任务、每本体结果和层级 bootstrap 置信区间。

---

## 8. 验收门与止损

### G0：OpenVLA baseline 门

- OpenVLA-OFT checkpoint、动作维度和控制接口通过；
- 六任务成功率可复现；
- rollout 保存完整且结果可重放。

失败：停止 critic 集成，先修策略 baseline。

### G1：数据门

- 每个训练任务有成功和失败；
- 失败不是环境初始化错误；
- 匹配任务×事件有足够独立 prefix；
- 确认集封存并记录哈希。

失败：不运行价值排序主表。

### G2：时钟门

- β 后验不在支持边界；
- ClockLNN 相对 β=0 在全部配对种子改善 duration；
- event MC value 不劣；
- 与 duration MLP/GRU-time 完成公平消融。

### G3：语义去塌缩门

- full-state 同事件 AUC超过事件常数基线；
- 增益在全新确认数据的置信区间下成立；
- failure-terminal 消融显著下降；
- 事件 token 打乱后性能符合预期；
- 测试至少达到预注册的独立 pair 数。

### G4：动作 critic 门

- B7 相对 B1/B6 改善候选动作排序；
- `Q−V` advantage 与真实动作后果一致；
- 没有明显 OOD 动作高估。

失败：保留 event-value auxiliary head，不进入闭环控制。

### G5：在线策略门

- critic-guided OpenVLA 的成功率或样本效率置信区间优于 B0；
- 至少大多数任务不劣；
- 延迟和计算预算可接受。

只有 G5 通过，才能写“完成 OpenVLA critic 跨本体迁移”。

---

## 9. 实施阶段

### Phase A：复现 OpenVLA-OFT

1. 确认 RLinf/RoboTwin commit和环境。
2. 确认 OpenVLA-OFT checkpoint。
3. 生成每任务×本体的 checkpoint、`unnorm_key` 和动作契约矩阵。
4. 先用 ALOHA 官方接口跑单任务 smoke test，再扩到六任务。
5. 对每个本体执行零动作、夹爪、单轴小动作和录制动作回放测试，通过 A0。
6. 固定动作归一化、chunk 和控制频率。
7. 输出 B0 成功率；明确采用 P-critic 还是 P-full。

### Phase B：OpenVLA rollout 数据

1. 源本体采匹配 seed 的成功/失败。
2. 每个初始状态重复 rollout。
3. shape probe 后保存池化 OpenVLA 特征、动作、proprio 和图像索引。
4. 用对象真值监督 `EventPosterior`，但线上输入屏蔽对象真值，通过 A1。
5. 自动提取事件和 failure-terminal。
6. 审计错误类型和数据划分。

### Phase C：OpenVLA shadow critic

1. 冻结 OpenVLA。
2. 训练 B1–B6。
3. critic 只旁路输出，不影响动作。
4. 通过 G2/G3 后再继续。

### Phase D：动作条件 critic

1. 增加 action chunk encoder。
2. 冻结候选生成协议并保证包含 OpenVLA 原始 chunk，通过 A2。
3. 构造真实或模型预测的候选动作后果。
4. 训练 B7/B8。
5. 通过 G4。

### Phase E：闭环在线评测

1. OpenVLA 生成名义 chunk，A2 冻结的 proposal 生成 N 个候选动作。
2. ETSF critic 排序。
3. 执行 top-1 或风险约束候选。
4. 比较 B0/B1/B7/B8。
5. 通过 G5 后解封确认集。

---

## 10. 当前证据

### 10.1 已完成：OpenVLA-OFT 单任务 baseline

`move_can_pot` 已在 RTX 4090 D 上完成 OpenVLA-OFT 端到端复现：

| 项目 | 结果 |
|---|---:|
| 模型 | RLinf OpenVLA-OFT RoboTwin SFT move_can_pot |
| 参数量 | 7.558B |
| 本体 | Piper 双臂，间距 0.6 |
| 协议 | demo_randomized |
| 动作 | 14 维，25-step chunk |
| 官方 eval seeds | 20 |
| 成功率 | **4/20 = 20%** |
| 平均 episode 长度 | 200 |
| BF16 模型显存 | 14.12 GiB |
| 合成输入完整 forward | 0.628 秒，输出 `(1,25,14)` |

固定版本：

- RLinf `a3816b596478dcd8a5c69a6ec1468c9519f77b5b`
- RoboTwin RLinf_support `0008ae6800df9f75fc8de7098bacb01735fd8fd2`
- RLinf OpenVLA-OFT `c9f0f3d31f438d98f6137936ea78a47f0b2ab087`
- transformers-openvla-oft `bc339d9ad707454c0c115970db43c260067c61ab`

运行环境通过镜像下载 checkpoint；单 GPU Ray worker 需要显式设置实际 Vulkan ICD 路径。FlashAttention/Pytorch3D 因系统 nvcc 12.4 与 PyTorch CUDA 13 不一致而跳过，当前 RGB-only eager/SDPA 路径已完成真实 rollout。

该结果只完成 B0 的一个任务。其余已有 checkpoint 的任务仍需串行复现；当前 RLinf checkout 没有 `adjust_bottle` 的 OpenVLA-OFT checkpoint/config，该任务只能改用 OpenPI baseline、另行训练 OpenVLA-OFT，或调整确认任务集。

### 10.2 已完成：OpenVLA×ETSF shadow 接线

已在同一 `move_can_pot` checkpoint和真实 Piper 环境中完成非控制 shadow 接入：

```text
OpenVLA action-prehidden [4096]
→ FactorizedEventTransport bridge [96]
→ semantic successor + ClockLNN
```

验收结果：

- 捕获 4096 维动作前 hidden；
- 逐 chunk 累积序列输入 ETSF；
- 8 个 action chunk、200 个环境步全部运行；
- shadow value/duration 输出均有限；
- hook 前后确定性 OpenVLA 动作逐元素完全一致；
- ETSF 参数量约 2.1 MB checkpoint，单卡资源充足；
- 当前 shadow 为固定种子随机初始化，只证明接线，不代表 critic 性能。

下一步不是让随机 shadow 控制动作，而是用 OpenVLA 自己产生的成功/失败 rollout保存 hidden、事件和失败末帧，训练 bridge/语义头/ClockLNN 后再做 shadow 离线门控。

### 10.3 已完成：对象状态机制实验

三本体留一、失败末帧、两折各 3000 步×5 种子：

| 留出本体 | 失败语义来源 | duration MAE：β=0→后验 | event MC MSE：β=0→后验 | success-only 同事件 AUC | 失败感知同事件 AUC |
|---|---|---:|---:|---:|---:|
| Piper | UR5 | 14.727→**4.162** | 0.026325→**0.024835** | 0.4333 | **0.6119** |
| UR5 | Piper | 18.066→**8.656** | 0.032681→**0.029241** | 0.4361 | **0.7917** |

能够支持：

- 可适配时钟分支有效；
- 失败终止状态此前被事件抽取器丢弃；
- 混合失败监督可以解除纯事件查找表退化；
- 失败表示可以迁移到完全留出的另一种机械臂。

不能支持：

- OpenVLA critic 已改善；
- OpenVLA 在线成功率已提升；
- LNN 必然优于 duration MLP/GRU；
- 已完成确认性跨本体 critic 迁移。

### 10.4 当前门控

```text
mechanism_gate_passed = true
decision_gate_passed = null
decision_gate_status = inconclusive_insufficient_matched_event_pairs
stop_before_critic_integration = true
```

`stop_before_critic_integration` 表示当前对象状态开发结果尚不足以直接进入在线控制；它不否定后续 OpenVLA shadow critic 实验。

---

## 11. 输出与版本管理

每个正式运行使用独立输出目录，不覆盖历史结果：

```text
manifest.json
config_frozen.json
data_hashes.json
checkpoint_hashes.json
training_history.csv
per_episode_predictions.csv
per_boundary_predictions.csv
per_action_candidate_predictions.csv
beta_posterior.csv
calibration.csv
gate_summary.json
report.md
models.pt
```

Git 仓库只提交：

- 代码；
- 配置；
- 文档；
- 不含敏感信息的小型汇总。

不提交：

- 原始图像和轨迹；
- checkpoint；
- 大型结果；
- 缓存和日志。

---

## 12. 下一步最小执行清单

1. 在 4090 服务器定位 RLinf 与 RoboTwin OpenVLA-OFT 环境。
2. 写入 RLinf、RoboTwin、模型和 processor 的 commit/hash。
3. 建立六任务×四本体的 checkpoint/动作契约矩阵，先通过 A0。
4. 复现 ALOHA 单任务 B0，再扩到具备 checkpoint 的任务；缺 checkpoint 的任务先标记阻塞，不伪造复现率。
5. 改造 rollout collector，替换 `task.play_once()` 为 `OpenVLAOFTPolicyAdapter`。
6. 保存池化 OpenVLA 特征、动作和失败终止状态；训练部署态 `EventPosterior` 并通过 A1。
7. 先训练 OpenVLA shadow B1/B2/B3/B6，不影响 actor。
8. G2/G3 通过后冻结候选生成器，通过 A2，再实现 B7 动作优势头。
9. 最后运行闭环候选动作排序和全新确认集。

一句话路线：

> 用 OpenVLA-OFT 负责看、听和行动；用失败感知共享语义头判断任务状态；用 ClockLNN 只校正不同机械臂的物理时钟；用显式事件传输生成 `V(s)`；再加入动作优势形成 `Q(s,a)`，最终以 OpenVLA 在线成功率而不是离线事件指标完成验收。

---

## 13. 模块设计依据与论文边界

本路线不是某一篇论文的原样复现，而是把现有方法组合成一个新的、可证伪的 ETSF 系统。下面区分“直接采用”“借鉴后改造”和“本项目自定义”。

| 模块 | 主要依据 | 采用关系 | 本项目中的变化 |
|---|---|---|---|
| OpenVLA 骨干 | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) | 直接采用预训练 VLA 思路 | 只作为冻结视觉—语言—动作骨干和特征源 |
| OFT 动作头 | [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645) | 直接采用 OFT 配方 | 保留并行解码、连续动作、action chunk 和 L1；critic 不改变 B0 动作头 |
| Action chunk | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) | OFT 与 ACT 都支持该设计动机 | ETSF 把整个 chunk 作为候选动作输入 critic |
| RoboTwin/RLinf 接口 | [RoboTwin](https://arxiv.org/abs/2409.02920)、[RLinf RoboTwin 官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) | 工程接口 | 本项目新增特征 tap、rollout schema 和 critic 旁路，不声称这些来自 benchmark 论文 |
| 跨本体表征动机 | [Open X-Embodiment](https://arxiv.org/abs/2310.08864) | 研究动机 | OXE/RT-X 证明跨机器人数据共享有价值，但不直接给出本文 critic 传输算法 |
| 语义 successor 头 | [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312) | 借鉴“可共享后继结构与任务/动力因素分离” | 本文预测规范事件后继可达性；不是标准线性 `Q=ψᵀw`，不能直接称为原版 successor features |
| HL-Gauss 价值分类 | [Stop Regressing: Training Value Functions via Classification for Scalable Deep RL](https://proceedings.mlr.press/v235/farebrother24a.html) | 直接采用分类代替标量 MSE 的训练思想 | support 仅由源验证集冻结；Stage 3 当前固定为 `[0,1]` |
| ClockLNN/CfC | [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) | 借鉴闭式连续时间/液态时间常数 | `ClockLiquidCell` 的 `aθ+β_b gθ` 是本项目低秩调制，不是论文或 `ncps` 的标准 CfC 层 |
| 显式事件传输 | [Between MDPs and Semi-MDPs](https://www.sciencedirect.com/science/article/pii/S0004370299000521)、[Per-Decision Option Discounting](https://proceedings.mlr.press/v97/harutyunyan19a.html) | 采用 SMDP 中随机持续时间与 `γ^D` 的理论 | `ρ_b` 可达率、`β_b` 后验及 `E[γ^D]` 组合为 ETSF 自定义分解 |
| Dueling critic | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) | 直接采用 `V+A` 分解 | `V` 被替换为 ETSF 传输值，`A` 对 action chunk 条件化 |
| 视觉 Q 与候选排序 | [QT-Opt](https://arxiv.org/abs/1806.10293) | 借鉴视觉闭环 Q 选择动作的思想 | 候选由 OpenVLA 名义 chunk 周围的受限 proposal 产生，不复现 QT-Opt 的完整 CEM/Q 学习流程 |
| 梯度冲突处理 | [PCGrad](https://arxiv.org/abs/2001.06782)、[CAGrad](https://arxiv.org/abs/2110.14048) | 仅作可选消融 | 先测梯度余弦，确有持续冲突才启用，不是默认组件 |
| 失败末帧、同事件排序、事件后验、`β/ρ` 少样本适配 | 无单一直接来源 | 本项目自定义 | 必须通过事件常数、failure-terminal、oracle event 和目标 TD 等消融验证贡献 |

### 13.1 创新表述边界

可以表述为：

> 本项目受 successor representation、连续时间网络、SMDP 折扣、价值分类和 dueling critic 启发，提出把跨本体 critic 分解为共享事件语义、低维本体时钟后验、事件可达率和动作优势的 ETSF 组合。

不能表述为：

- “ClockLNN 就是 CfC 论文原模型”；当前代码是低秩时间调制的定制 cell。
- “语义头就是标准 successor features”；当前目标是事件后继/MC 价值，不满足原论文的完整线性分解。
- “OpenVLA-OFT 原论文自带跨本体 critic”；OFT 论文解决的是 VLA 微调与动作解码，ETSF critic 是本项目新增。
- “RLinf 已支持本文全部机械臂和任务”；只有实际通过 A0 的 task×body 配置才能进入该表述。
- “重复运行确定性 OFT 得到 N 个候选”；候选分布必须由 A2 明确定义。
