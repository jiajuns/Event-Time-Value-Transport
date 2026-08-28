# ETSF 跨本体共享头：最新设计、训练监督、迁移与证据协议

> 版本：2026-08-28 r12 设计与实现对齐版。本文以当前 Source63 因果历史训练器、Piper embodiment adapter、Formal190 完整外层嵌套选择、严格 rank 数值合同和 evaluation400 v4 独立审计基础层为准。本文严格区分“代码已经实现并通过合成回归”“已经挂到远端但尚未开始参数更新”“仍只是协议基础、尚未接入 runner”和“尚未获得的真实成功率结论”。

当前结论先写在前面：共享头已经是一个可训练、可迁移、接口上可插拔的动作条件事件模型，但“跨本体后提高任务成功率”仍是待验证假设，不是已经得到的实验结论。Source63 尚未开始 GPU 参数更新，Piper 目标 development300 尚未获准采集，Formal190 与 evaluation400 也尚未执行。

## 1. 一句话定义

这里的“共享头”不是简单把一个分类器复制到不同机器人上，而是：

1. 用同一个 **动作条件事件世界模型核心** 表示跨本体共同的事件转移规律；
2. 用一个 **同状态、基线相对的动作排序残差头** 学习“换一个动作候选会不会更好”；
3. 为新本体只训练低容量 state/action/clock/body adapter；
4. 用目标本体监督数据重新校准六个预测头和候选接受门；
5. 用完整外层嵌套的 Formal190 估计“选择规则本身”的泛化增益；
6. 只有配对任务成功率的 bootstrap 下界严格为正时，才允许插件替换 actor 默认动作。

因此它是“共享事件因果结构 + 本体小适配器 + 目标监督校准”的迁移方法，不是把 Aloha 的原始关节动作零样本发送给 Piper。

## 2. 总体数据流

```text
同一分支的因果状态历史 H_t（最多 8×960-D，mask 后得到 z_t）
+ proprio q_t（14-D）
+ 当前事件 e_t（e0/e12/e3/e4/eK）
+ 当前谓词 p_t（moved/lifted/near_goal/stationary/success）
+ 同一根状态下的候选动作块 u_t^0 ... u_t^K（H×14）
+ body_id / policy_id
        │
        ├── StateAdapter：目标本体的低秩状态残差
        ├── ActionAdapter：目标本体的对角 + 低秩动作残差
        └── 共享 ActionConditionedEventWorldModel
                │
                ├── p(post-event)
                ├── p(next-reached-event)
                ├── p(duration D | current event)
                ├── p(final success)
                ├── p(recovery | operational regress)
                ├── p(object-state delta)
                └── source_contract_rank_score（候选排序量，非概率）
                         │
                五成员 ensemble + 六头全局证据门
                         │
                初始 e0 根状态组合五个适用头的不确定性
                         │
                Formal190 冻结 margin/uncertainty guard
                         │
                通过则执行替代候选，否则回退最低合法 baseline
```

actor（SmolVLA/OpenVLA/其他策略）保持冻结。模块只读取候选并返回“选哪个”，所以在接口层面是可插拔 critic；但每种策略和本体仍必须有匹配的 state/action contract、因果历史构造器、本体动作 adapter 与目标监督校准。

## 3. 输入表示为什么能跨本体

### 3.1 共享状态不是 action-expert hidden

SmolVLA 路径使用 flow denoise 之前的 VLM prefix 最后一个有效 state token，固定为 960-D。这个状态只由图像、语言和机器人当前状态决定，同一根状态的不同 flow-noise 候选必须逐位相同。

旧的 720-D action-expert hidden 会随候选噪声变化，已经被明确禁止作为“候选共享状态”。否则模型能从 hidden 中偷看到候选身份，无法形成干净的同状态动作干预。

### 3.2 共享的是事件空间，不是关节坐标

规范事件词表为：

```text
e0, e12, e3, e4, eK
```

规范动态谓词为：

```text
moved, lifted, near_goal, stationary, success
```

不同本体的观测和动作先经过各自 adapter，再进入共享 96-D semantic/action-effect 空间。事件定义、对象相对关系和最终成功语义可共享；关节尺度、动作维度、控制周期和执行速度不能直接共享。

### 3.3 最新因果历史合同

共享状态编码不再把每次 query 当作彼此无关的单步样本。r12 固定合同为：

```text
format = etsf_same_branch_causal_hidden_history_v1
max_history_steps = 8
input = same_branch query_hidden[:current_query + 1]
truncation = 左侧截断，只保留最近 8 步
padding = 右侧精确零填充，mask 为连续 True 前缀
future hidden = 禁止
cross-branch / cross-group history = 禁止
root effective history = 1 步
```

设当前 query 索引为 (j)，未截断前缀为

\[
\bar H_j=[h_0,h_1,\ldots,h_j].
\]

模型只保留

\[
H_j=\operatorname{tail}_8(\bar H_j),
\]

并在右侧补零到固定长度 8。未来的 (h_{j+1:}) 无论如何变化，都不能改变 (H_j)。Source 的 future-latent 辅助目标使用“当前因果前缀 + 当前 `query_post_hidden[j]`”；它不会把下一次 query 的 hidden 偷放进当前输入。

根状态下所有候选共享同一 pre-action hidden，有效历史严格为一步，因此 r12 的根候选语义表示与旧单状态实现逐位一致。Piper Schema6 有同分支 `branch_hidden`，所以输入历史也按上述规则构造；但它没有 `query_post_hidden`，因此明确不训练、也不伪造 post-latent 目标。

该合同解决的是训练时的时序泄漏与 GRU 重置问题，不等同于已经完成在线事件观察器。当前 v3 simulator runner 仍依赖环境 pose 推导谓词，尚不能据此声称真实机器人上已具备非特权、多步事件跟踪。

## 4. 共享事件核心

令因果状态历史及其 mask 为 (H_t,M_t)，状态语义为

\[
z_t = E_s(H_t,M_t),
\]

动作效果为

\[
a_t = E_a(u_{t:t+H}, b, \pi),
\]

其中 `E_s` 是带历史 mask 的 GRU semantic encoder，`E_a` 是保留动作顺序的 temporal GRU，并由 `body_id`、`policy_id` 的 FiLM 参数调制。共享转移表示为

\[
\tau_t = F\big[z_t, a_t, z_t\odot a_t, E_q(q_t), E_e(e_t), E_p(p_t)\big].
\]

当前 SmolVLA-native 配置为：

| 项目 | 当前契约 |
|---|---:|
| state 输入 | 最多 8×960-D，根状态有效 T=1 |
| 动作 | H×14，带 step/feature mask |
| proprio | 14-D |
| semantic/action-effect | 96-D |
| transition hidden | 128-D → 96-D |
| clock hidden | 48-D |
| 对象效果 | moving object 的 xyz，3-D |
| body registry | Aloha row 0，预留 Piper row 1 |
| policy registry | SmolVLA row 0，预留 OpenVLA row 1 |

当前 SmolVLA-native 共享核心约 38.26 万参数，其中 semantic encoder 约 15.96 万、action encoder 约 8.24 万、transition trunk 约 7.62 万、clock cell 约 2.09 万、当前复合 residual rank MLP 约 1.90 万。参数量只是实现规模说明，不是效果证据。

Piper 仍使用 SmolVLA policy row 0；预留的 OpenVLA row 1 不会因为名称相似而被误用。

## 5. 六个正式预测头

| 头 | 预测目标 | 监督与基线 | 正式用途 |
|---|---|---|---|
| post-event | 执行动作块后的事件 | 事件分类；对比 current-event persistence | 判断即时事件效果 |
| next-event | 下一次真正到达的事件 | 只用 duration-observed 行；对比 train-fold reachable prior | 判断后续事件转移 |
| duration | 距下一事件的决策步数 | `log1p(D)` 对数正态 + 右删失；对比 current-event median/lognormal | 时间预测与风险 |
| success | 分支最终成功 | factual base success logit；对比训练折 prevalence | 成功概率校准，不直接承载 rank residual |
| recovery | 发生 operational regress 后是否恢复 | 独立二分类，只在 observed regress 行监督 | 失败恢复风险 |
| object effect | 对象 xyz 变化 | Gaussian mean/scale；对比 zero 和 robust median | 对象状态变化预测 |

模型内部还保留 reach、post-predicate 和 future-latent 等辅助输出，但它们不冒充上述六头的正式通过证据。

六头的“适用性”不是统一的。初始 pre-action 根状态适用 post-event、next-event、duration、success 和 object-effect；recovery 只有在已经观察到 operational regress 后才有定义。因此 root 决策使用五头数值不确定性，但系统级授权仍要求 recovery 在其适用样本上通过 support、performance 和 uncertainty 三类门。

训练目标也不能互相替代：事件头用分类目标，success/recovery 用二分类目标，duration 用带右删失项的 `log1p(D)` 分布目标，object-effect 用带尺度的 Gaussian NLL。把这六项压成一个“总 loss 下降”不能证明每个头都有效，所以校准阶段逐头与简单基线比较。

### 5.1 时钟与事件核心的硬隔离

本体速度参数 `beta` 和 `dt` 只进入 clock cell。clock 读取 `transition.detach()`，因此 duration 监督不能反向改写共享事件几何，`beta` 也不能改变事件、成功或对象预测。这是“共享事件结构、单独适配执行速度”的关键边界。

### 5.2 Recovery 头为什么单独训练

Recovery 的语义是

\[
p(\text{recovery}\mid\text{operational regress}),
\]

不是普通三分类中的一个随意类别。当前实现用独立线性 head 读取 `transition.detach()`，使用独立 optimizer；其梯度既不更新共享核心，也不更新 Piper adapter。只有回退持续至少三个保存状态，并且后续恢复旧 peak 持续至少三个状态或到达 `eK`，才构成监督样本。

## 6. 最新共享动作排序头

这是本轮最重要的修复。

### 6.1 同根候选与精确 baseline 锚定

对同一 simulator 根状态，设最低合法候选为 \(u^0\)，其他候选为 \(u^i\)。共享头只看动作效果差：

\[
\Delta a_i=a_i-a_0,
\]

\[
r_i=h([\Delta a_i, z_t\odot\Delta a_i])-h(0).
\]

因此 baseline 的残差严格为 0，场景本身“难不难”留给 factual success/event heads，动作头只学习“相对默认动作发生了什么变化”。候选可以重排存储，但每个 logical group 必须有且仅有一个 baseline，且其 original candidate index 必须是该组最低合法索引；跨组锚定、重复索引或缺 baseline 都直接报错。

### 6.2 Source63 实际训练契约

r12 Source63 计划沿用以下训练开关：

```text
action_rank_residual = true
action_rank_success_only = false
freeze_factual_core = false
```

所以 residual 训练时不是简单加在成功概率上，而是加在一个复合科学排序量上。Piper 必须精确复现：

\[
B_i=\frac{\ell_i^{success}}{T}
+0.25\sum_k p_i(e_k)v_k
-0.05\frac{\exp(\mu_i^D)-1}{s_D},
\]

\[
R_i=B_i+\frac{r_i}{T}.
\]

其中：

- 当前 r12 Source63 计划的初始 \(T=1\)，但模型文件仍逐成员保存原始 `success_temperature`，不能由下游假设或覆盖；
- 事件值固定为 `[0, 0.25, 0.5, 0.75, 1]`，顺序对应 `e0/e12/e3/e4/eK`；
- \(s_D\) 是每个 Source member 自己 checkpoint 中、由其训练数据得到的 `duration_scale`；不能使用 ensemble 平均值；
- `R_i` 在代码中叫 `source_contract_rank_score`。

最重要的语义约束是：

```text
source_contract_rank_score 不是 success_logit
source_contract_rank_score 不是 success_probability
```

factual `success_logit` 始终保持原值，继续用于成功 BCE 和目标本体的概率校准。早期把 `base_success_logit + residual` 直接写回 `success_logit` 的路径已经从 Piper 生产接口移除。

### 6.3 IEEE float32 数值契约

训练、Formal190 复算和 evaluation400 在线选择现在共同绑定以下字符串：

```text
ieee754_float32_training_order_base_plus_residual_div_temperature
```

它不是说明性注释，而是每个 Source member 已签名 `source_rank_score_contract` 的组成字段。精确运算顺序为：

```python
base32 = as_float32(base)
residual32 = as_float32(residual)
temperature32 = as_float32(raw_member_temperature)
scaled32 = residual32 / temperature32
composite32 = base32 + scaled32
```

只有 `composite32` 产生以后，五成员均值和 margin 才允许提升为 float64。生产链拒绝以下漂移：输入三矩阵是 float64、先在 float64 计算再降精度、交换加除次序、用容差 `allclose` 接受结果、或 composite 相差一个 float32 ULP。测试专门覆盖 `T=0.3`、`0.1f+0.2f`、1-ULP 篡改和 float64 promotion。

### 6.4 五成员权威不是下游自报字段

Formal190 从五个已经验证的成员 contract 按 `member_index=0..4` 派生有序权威：

```text
source_rank_member_authority = {
  source_rank_numeric_contract,
  members: [
    {member_index,
     source_checkpoint_file_sha256,
     source_rank_score_contract_sha256,
     success_temperature}, ...
  ]
}
```

该对象的 canonical SHA 同时绑定在 root ranker、calibration、ensemble manifest、calibration receipt、identity bridge、paired core、selector authority、condition runner 和 sealed result evaluator 中。成员重排、温度漂移、source checkpoint 替换、contract SHA 替换或只在下游重新签名，都会 fail-closed。这样 Formal190 选出的阈值与在线真正执行的排序公式保持逐位一致。

### 6.5 为什么这种排序量可能改善成功率

排序量本身不是概率，但它可以作为“候选偏好统计量”。是否真的改善成功率，不由它的数值名称决定，而由目标本体的配对监督决定：在 Formal190 中比较“按冻结规则接受候选”和“执行最低合法 baseline”的真实最终成功结果，再冻结 margin/uncertainty guard。只有 selection-aware 外层 OOF 的 bootstrap 增益下界严格大于 0，才允许在完整 190 组上拟合最终 deployment rule；最终效果仍要由 evaluation400 回答。

## 7. Piper 跨本体 adapter

Source 核心训练完成后，Piper 不重训整个世界模型。冻结项包括：

- 所有共享事件、成功、时长、对象和动作排序参数；
- Aloha body row 0；
- SmolVLA policy row 0；
- 预留 OpenVLA policy row 1；
- 每个 Source member 的 rank-score contract 与 checkpoint SHA。

默认仅训练：

1. `StateAdapter`：960-D identity + rank-16 低秩残差；
2. `ActionAdapter`：14-D identity diagonal + rank-4 低秩残差；
3. `clock_beta` 与 `clock_log_step_scale` 两个标量；
4. 预先保留的 Piper body embedding row 1；
5. 另行训练、stop-gradient 的 conditional recovery head。

默认有效目标参数约 30,864 个，远小于共享核心。每次 optimizer step 后都会逐张量验证：除 Piper body row 外，Source core 必须 bit-exact 不变。

r12 还把因果历史合同作为 Source checkpoint 的强制身份字段。Piper 训练器只有在 Source checkpoint 的 `causal_history_contract` 与本地精确对象完全一致时才加载；缺字段的旧 checkpoint 即使张量形状兼容也会 fail-closed。当前 Source 合同 canonical SHA 为：

```text
de846ddf7a32d5f27ec775eb00fb81f4a3f21652948b6168d3b78c957e2ffb0e
```

Piper 对该合同的应用声明另外绑定：输入使用 Schema6 同分支因果前缀，但因为目标 schema 没有 `query_post_hidden`，post-hidden supervision 为不适用，而不是用零张量冒充观测。该 application contract SHA 为：

```text
64cfa85e55c608b584b32cd4ceddac0375021ab532c915dc257cdd7ba2ec3d72
```

### 7.1 消除轨迹长度偏置

最新训练不再把所有 transition 行直接 shuffle。那种做法会让长轨迹获得更多 optimizer 权重，也会重复放大每行相同的 terminal success 标签。

现在 dense sampler 每个 epoch 对每个 logical group 恰好抽取一个 transition，组内按固定 seed 的循环排列逐步覆盖不同 transition。因此长轨迹能提供更多状态多样性，但每个 epoch 的组权重与短轨迹完全相同。validation、recovery prior 和 recovery validation 也按 logical group 等权；duration/object/recovery 的行级 observation mask 保持不变。

候选 ranking loss 同样先在组内计算，再对 logical group 等权平均，不把同一组的候选对当作独立样本。

## 8. 五成员 ensemble 与六头校准

计划冻结五个不同训练 seed 的 Source/Piper 成员。目标本体校准严格按 logical group 做五折 cross-fit，不能把同一轨迹的行拆到不同折。

每个头必须同时满足三类条件：

1. **support gate**：正/负、观测/删失或各事件类别拥有足够独立 group；
2. **performance gate**：相对简单基线的 group-bootstrap 增益下界不小于 0；
3. **uncertainty gate**：不确定性对误差具有有效排序能力，AURC/秩相关门通过。

当前最低独立组阈值为：post/next/duration/recovery 每侧 10 组，success 每类 50 组，object nonzero/near-zero 每侧 50 组。任一正式头失败，则“六头主路径”整体不授权。

校准方式为：

- post/next-event：五折 temperature scaling；
- success/recovery：五折 binary temperature scaling，检查 NLL、Brier 和 AP；
- duration：五折 scale multiplier，检查 MAE、NLL 和 90% coverage；
- object：五折 Gaussian scale multiplier，检查 zero/robust-median L2；
- 不确定性：ensemble epistemic + 单成员 aleatoric，总量归一化后按适用头组合。六个头都必须通过全局证据门；但初始 pre-action `e0` 根状态尚未观察到 operational regress，因此在线 root uncertainty 只平均 post-event、next-event、success、duration、object-effect 五个适用头。recovery 不进入该根状态数值，但其全局 support/performance/uncertainty gate 仍必须通过。

这里的 success 校准只使用 factual base `success_logit`；动作排序使用 `source_contract_rank_score`。两条路径必须在 schema、命名和 SHA 上完全分离。

### 8.1 两层 cross-fit 不能混为一谈

当前协议有两个统计层次：

1. **六头预测质量层**：按 logical group 做 cross-fit，判断每个头相对简单基线是否有可用预测能力；
2. **root 选择规则层**：评估“先校准、再搜索 margin/uncertainty gate”这一整条选择算法在未参与选择的组上是否仍有增益。

第二层不能直接复用“在全 190 组上拟合好的温度/尺度和阈值”，否则外层 heldout 标签会通过校准参数间接泄漏回选择过程。r12 对每个外层 fold 执行完整隔离：

```text
Formal190 = 5 folds，每折 152 train + 38 heldout

仅用该折 152 train：
  拟合 post / next / success temperature
  拟合 duration / object scale
  拟合 object robust scale
  计算 train quality、threshold 与 20 个 root grid candidates
  选择该折唯一规则

用完全相同的 train-fitted 参数：
  推理 38 heldout
  应用已选规则一次
  记录 heldout paired success / harmful change

拼接 5 折：
  每个 Formal group 恰好作为 heldout 一次
  用固定、共享的 logical-group bootstrap draws 估计最终 LCB/UCB
```

外层 heldout 标签不能用于温度、尺度、robust baseline、uncertainty normalization、阈值或 grid 选择。任一 fold 无法拟合所需参数、没有满足 support 的候选或无法选择规则，primary 直接失败，不允许用其他 fold 或全数据参数补洞。

实现中的关键 fail-closed 字段为：

```text
root_outer_nesting_contract.outer_heldout_labels_used_for_any_parameter_or_selection = false
root_outer_nesting_contract.complete_root_pipeline_outer_nesting = true
upstream_predictions_already_group_crossfit = true
complete_temperature_scale_and_root_double_nesting = true
selection_aware_oof_evidence.passed_for_primary = true
```

只有这些字段、六头全局门和 full190 deployment candidate 同时成立，才允许生成最终 deployment selector。外层 OOF 回答的是选择流程是否泛化；通过以后再用完整 190 组拟合一个最终规则，供完全未见的 evaluation400 使用。

## 9. Formal190 候选接受门

目标 development300 预注册划分为：

```text
80 train + 30 internal validation + 190 formal selector calibration
```

Formal190 是监督数据，用于冻结候选接受规则，不是最终 evaluation400。每个 root group 都保存最低合法 baseline、所有合法候选和各自最终 success。

最新 selector 设计应使用五成员的 baseline-relative `source_contract_rank_score`：

\[
\Delta R_i=\frac{1}{5}\sum_m(R_{m,i}-R_{m,0}).
\]

执行候选必须同时满足：

```text
候选是非 baseline 中 ΔR 最大者
ΔR > frozen margin
candidate/baseline 五个根状态适用头的结构化不确定性 <= frozen maximum
六个预测头的 support/performance/uncertainty gate 全部通过
```

固定搜索网格为：margin `{0, 0.1, 0.25, 0.5, 1.0}`，maximum uncertainty `{0.25, 0.5, 0.75, 1.0}`。候选规则还必须满足：

- changed groups 至少 50；
- success-discordant groups 至少 20；
- 每个五折子集 changed 至少 10、discordant 至少 4；
- paired success gain 的 logical-group bootstrap 95% LCB 严格大于 0；
- 已执行改变中的 harmful rate，其 bootstrap 95% UCB 不超过 0.10。

选择优先级为：最大 gain LCB → 最大 selected success rate → 最小 harmful UCB → 最大 coverage → 更保守的门。LCB 等于 0 只表示非劣，不足以启用 primary rerank。

这里必须报告两组结果：selection-aware 外层 OOF 结果，以及通过后使用全 190 拟合出的 deployment rule。不能在同一 190 组上搜索 20 个候选规则，然后把最优规则在这 190 组上的 LCB 当作未见泛化证据。

当前 adapter 已输出严格的 composite rank score，calibrator/bridge/post/paired 链已加入完整外层隔离字段并通过合成回归。正式 production authority、经独立批准的真实 execution inventory 和真实目标本体证据尚未生成，因此生产执行仍保持 fail-closed，不能声称已经提高任务成功率。

## 10. 在线可插拔接口

生产插件需要的最小逻辑接口是：

```python
prediction = ensemble.predict_grouped_candidates(
    shared_state=state_960,
    current_event=current_event,
    current_predicates=current_predicates,
    proprio=proprio_14,
    candidates=actions_h14,
    logical_group_id=group_id,
    candidate_index=candidate_indices,
    baseline_mask=baseline_mask,
)

decision = frozen_selector.select(
    source_contract_rank_score=prediction.rank_score,
    root_applicable_five_head_uncertainty=prediction.structured_uncertainty,
)
```

插件必须验证：五个 adapter SHA、五个 Source checkpoint/rank contract SHA、event spec、校准 artifact、runtime authority、candidate identity 和 baseline identity。缺任意字段、字段命名仍是 `adjusted_success_probability`、候选不共享同一根状态，或本体/策略 calibration 不匹配，都回退 baseline。

因此它可以适配多种算法，但不是“随便接上就有效”：

- VLA/自回归策略：多次采样得到候选；
- diffusion/flow 策略：不同噪声得到候选；
- ACT：不同 latent/sample 得到候选；
- 单输出策略：只能使用事先验证过的受限扰动候选。

每种算法都要提供同一时刻的候选共享状态、合法动作 mask、fallback candidate 和内容寻址的 policy adapter。

### 10.1 “可插拔”实际要求的 adapter 合同

一个新 actor 不是只实现一个 Python 函数名就算迁移完成，至少要满足：

| 接口项 | 必须提供的内容 | 失败处理 |
|---|---|---|
| state adapter | 当前观测对应的策略共享特征及同分支因果历史 | 回退 baseline |
| action adapter | 候选动作块、时间 mask、feature mask、原始单位定义 | 回退 baseline |
| root identity | 所有候选来自同一 simulator/real-world snapshot | 整组拒绝 |
| candidate registry | 唯一 candidate index、唯一最低合法 baseline | 整组拒绝 |
| body/policy identity | 内容寻址的 checkpoint、body row、policy row | 拒绝不匹配 authority |
| event observer | 当前事件/谓词及适用性，不可读取未来 | 不确定则 abstain |
| calibration | 该本体、该策略、该任务分布上的冻结 artifact | 回退 baseline |

因此，“适配多种算法”是接口层成立、效果层需要逐算法验证。对 diffusion/flow，可以使用不同噪声候选；对自回归 VLA，可以多次采样；对 ACT，可以用不同 latent/sample；单输出策略若没有自然候选，只能使用预先证明合法的局部扰动，不能任意制造动作。

### 10.2 当前在线观察器的真实边界

目前训练数据里的事件/谓词监督来自 simulator object pose 和 proprio 轨迹。当前 evaluation400 v3 runner 的根语义是固定 `e0`，并使用 runtime snapshot 的 object pose 重建谓词；这是一条 simulator privileged-state 上界路径，不是已经完成的视觉在线观察器。当前 root-only 候选协议与 `e0` 自洽，但它不能证明多步、非特权、真实机器人事件跟踪。

面向真实部署的下一版观察器应只读取 actor 可见信息：图像/VLM state、proprio、历史预测与执行回执，输出

```text
p(e_t), p(predicates_t), observer_confidence, applicability_mask
```

并维护同分支因果 hidden 队列。置信度不足、事件分布矛盾或 observer 合同不匹配时必须回退 baseline。这个 observer 还没有接入当前在线 runner，因此论文现阶段应把“跨本体共享世界模型”和“非特权在线事件识别”作为两个独立实验问题。

### 10.3 evaluation400 v4 的四条件归因设计

为区分“多采样”“success head”“复合 rank”和“完整安全门”的贡献，v4 计划对每个 pair 固定执行四个条件：

| 条件 | 动作规则 | 用途 |
|---|---|---|
| baseline | 最低合法 actor 候选 | 主对照 |
| success_only_guarded | 只按 calibrated factual success，保留 guard | 检验 success head |
| composite_rank_ungated | 按复合 rank，总是接受最高分 | 检验 rank 信号及无 guard 风险 |
| etsf | 复合 rank + margin + uncertainty + 全局门 | 完整方法 |

400 个 pair 对应严格 1600 个 condition rollout。primary comparison 为 `etsf - baseline`；其他比较用于归因并做多重比较校正。当前 v3 runner 只覆盖 baseline/ETSF 主路径，四条件 v4 尚未接入 simulator runner 或 external executor。

### 10.4 六头预测的“先承诺、后解封”审计基础层

r12 新增了独立纯模块 `smolvla_piper_evaluation400_audit_contract_v1.py`，用于防止跑完以后依据真值重写预测：

1. 每个 root 在执行前提交五个适用头的五成员原生 C-contiguous float32 tensor、逐 tensor SHA 和 tensor-set SHA；一 ULP 变化也会被识别；
2. root precommit 绑定 snapshot、全部合法候选、五个成员 checkpoint/rank contract、数值合同、校准、不确定性、event spec 与 runtime authority；
3. recovery 在 root 固定 `not_applicable`；一旦后续出现可判定 regress 的执行阶段，必须在每个可能 continuation step 前提交五成员 recovery logit；
4. broker 强制 sequence、step、previous SHA 和 ACK，拒绝重放、错步、断链及未确认后继续执行；
5. target 使用 X25519、HKDF-SHA256 和 ChaCha20-Poly1305 密封，AAD 精确绑定 core/pair/condition/root/runtime；
6. 只有 `1600 conditions + 400 pairs + 0 retry + 0 incomplete + 0 exclusion` 全部满足后才允许解封；
7. 六头指标先在 pair 内等权，再以 `pair_id` 为 cluster bootstrap 单元，显式报告 applicable、observed、censored/missing 与 insufficient support。

这一模块当前只是 v4 的 schema、密码与统计基础层：它不能启动 simulator、不能写 WORM ledger、不能执行 selector，也尚未与 v3 runner/executor 集成。因此“审计合同测试通过”不等于“evaluation400 已经执行”。

## 11. 训练数据和监督是否必要

需要监督数据。当前标签来自同一根状态下实际执行各候选分支后的轨迹：

- 对象 pose/proprio 轨迹推导 current/post/next event 与动态谓词；
- 首次到达下一事件的步数监督 duration，未到达时做右删失；
- 分支终局 success 监督 factual success 和组内候选排序；
- 持续回退/恢复轨迹监督 conditional recovery；
- 对象前后位置差监督 object effect。

Source63 为 Aloha-AgileX + SmolVLA + `move_can_pot` 的 63 个 logical group，冻结划分是 44 train / 14 validation / 5 development holdout，五个训练 seed。它不是凭空生成的“官方通用事件数据集”，而是由现有策略在任务环境中的监督分支轨迹构成。

“官方数据”需要分三层回答：

| 层次 | 当前使用内容 | 是否是官方现成 ETSF 标注 |
|---|---|---|
| actor 初始化 | SmolVLA/OpenVLA 既有策略 checkpoint | 是既有模型资产，但不是事件六头标签 |
| Source63 | 在 Aloha-AgileX、`move_can_pot` 上对 SmolVLA 候选做同根分支执行 | 否，是本项目采集并派生的监督数据 |
| Piper target | development300 的目标本体分支轨迹 | 尚未采集，且不是官方通用数据集 |

普通成功/失败终局标签只能训练 success 和一部分 ranking，不能完整训练 duration、recovery、object effect 与 action-conditioned event transition。六头完整监督至少需要时间序列、事件/谓词派生所需的对象状态、动作候选身份、删失标志和分支终局结果。

Source63 的 63 个 logical group 对六头世界模型仍然很小，所以当前设计用四种方式控制过拟合：共享核心参数量受限、同根相对残差消除场景难度、按 logical group 等权采样/切分、五成员 OOF 与目标本体低容量 adapter。它们只能降低风险，不能把小数据自动变成充分证据。是否数据不足要由每头 support gate、学习曲线和扩量后的 OOF 指标判断，而不能仅凭训练 loss。

Piper 的最新 development300 方案是目标监督适配与 selector 校准数据；预注册已经冻结，但当前 `collection_authorized=false`，尚未采集。最终 evaluation400 在所有模型、校准和 selector 冻结后才可读取，用于回答“是否真实提升任务成功率”。

### 11.1 从原始轨迹到六头监督

```text
同一 root snapshot
  ├── baseline candidate 执行轨迹
  ├── candidate 1 执行轨迹
  ├── candidate 2 执行轨迹
  └── candidate 3 执行轨迹
          │
          ├── query hidden / action chunk / proprio
          ├── object pose sequence → predicates / event / xyz delta
          ├── first reached event → duration + observed/censored
          ├── terminal outcome → factual success
          └── persistent drop then restoration → conditional recovery
```

划分单位始终是 logical group，而不是 transition 行或 candidate 行。属于同一 root、同一轨迹或同一 resolved seed 的关联样本不能跨 train/validation/heldout 泄漏。

## 12. 论文中可以突出的创新

可以把贡献写成以下五点，而不是泛泛地称为“又一个世界模型”：

1. **事件级跨本体因果接口**：共享的是可逆事件、持续时间、恢复和对象效果，不要求不同机器人共享原始动作坐标或执行时钟。
2. **同根、baseline-relative 动作效果头**：用真实 simulator root intervention 隔离场景难度与动作差异，并保证 baseline residual 恒为零。
3. **概率预测与科学排序量解耦**：成功概率单独校准，复合 rank score 不伪装成概率；最终以目标本体真实 paired success 决定是否启用。
4. **因果历史与时钟隔离的迁移合同**：共享状态只能使用同分支过去，根候选保持同状态干预；本体速度只进入 stop-gradient clock，使事件几何和执行节奏可分开迁移。
5. **六头全局证据门 + selection-aware 安全选择器**：事件、时间、成功、恢复、对象变化全部经过 support/performance/uncertainty 门；初始 `e0` 只组合当时可适用的五头；完整外层嵌套评估“校准 + 阈值搜索”整条选择算法，而不是在同一 Formal190 上选择并报喜。

工程上的附加创新是内容寻址和 fail-closed 证据链：模型、adapter、selector、runner 和结果 artifact 都绑定精确 SHA，避免训练公式与在线执行公式悄悄漂移；evaluation400 v4 还引入执行前预测承诺、recovery step 链和完成后统一解封。后者目前是已测试的基础层，不能在论文中写成已完成正式实验。

## 13. 与两篇最相近工作的边界

### 13.1 相对 VLAC

[VLAC](https://arxiv.org/abs/2509.15937) 的核心是从两帧图像和语言目标预测有符号 progress delta 与 done，并把这个稠密过程奖励放进真实机器人 PPO；它也展示了跨数据集、跨场景和多机器人并行学习。它与本文最相近之处是“过程 critic 可以比动作策略更容易迁移”。

本文不能把“跨本体 progress critic”本身当成独有创新。真正的区别应写成：

- VLAC 主要比较已经发生的两帧过程变化；本文在同一未执行根状态上显式比较多个候选动作块，学习 `p(outcome | e_t, u_t)` 与 baseline-relative 动作效果；
- VLAC 输出标量 progress/done；本文分解为 post-event、next-event、duration、success、conditional recovery、object effect，并保留删失、适用性 mask 和每头证据门；
- VLAC 用 reward 驱动 PPO 更新 actor；本文保持 actor 冻结，作为可插拔的候选选择器接入 flow/diffusion/ACT/自回归策略；
- 本文不把“在多台机器人上共同训练”直接等同于“跨本体机制迁移”，而是用 Source checkpoint、目标低容量 adapter、LOBO 和目标本体 paired task success 分开验证。

### 13.2 相对 ProgressVLA

[ProgressVLA](https://arxiv.org/abs/2603.27670) 已经包含 action-conditioned world model、可迁移 latent action、progress estimator 和对 diffusion sampling 的可微 guidance；它用规范化时间 `t/T` 作为 progress 监督，并在 LIBERO、CALVIN 与真实机器人上评测。因此，“世界模型预测候选未来 + progress 引导动作”也不能单独作为本文创新。

本文应突出的是不同的问题设定与证据契约：

- 不把任务压成单调标量时间进度，而是建模允许停滞、回退和恢复的离散事件图；
- 对 duration 使用事件条件分布与右删失监督，对 recovery 使用 observed-regress 条件监督，对 object effect 使用物理量监督；
- 不通过梯度改写某一种 diffusion sampler，而是对任意 actor 已产生的合法候选做同根反事实排序；
- 将事实成功概率与复合动作排序统计量严格分开，避免把 heuristic rank 当作 calibrated probability；
- 是否启用替代候选由目标本体 Formal190 的真实 paired success LCB 和 harmful-rate UCB 冻结，再在完全未见的 evaluation400 上检验。

最稳妥的论文定位不是“第一个 progress world model”，而是：**面向跨本体、异构 actor 的事件结构化候选效果模型，以及带适用性、不确定性和配对任务成功证据的安全插件化选择协议。**

## 14. 必须报告的 baseline 与消融

正式实验至少包含以下组，否则无法证明增益来自哪一部分：

| 组别 | 定义 | 回答的问题 |
|---|---|---|
| Actor-only | 原策略最低合法 baseline，不调用 ETSF | 原始任务成功率是多少 |
| Random candidate | 在同一合法候选集合随机选 | 多采样本身是否带来增益 |
| Success-only guarded | 仅按 calibrated factual success 排序，保留接受门 | 共享动作残差是否必要 |
| Composite rank ungated | 使用完整复合 rank，但总是选最高分 | guard 是否降低有害替换 |
| Scalar progress critic | 只预测单一 progress/done | 事件分解是否优于标量进度 |
| Action-conditioned scalar | 世界模型只预测候选 progress | 六头结构监督是否必要 |
| Event model without action | 只预测当前轨迹事件，不看候选差异 | 改善是否真正来自动作条件 |
| Ours without relative anchor | 去掉同根 baseline-relative residual | 根状态难度消除是否有效 |
| Ours single-state | continuation 每步重置 semantic GRU | 因果历史是否改善多步预测 |
| Ours without target adapter | Source 直接零样本用于 Piper | 低容量跨本体适配是否必要 |
| Ours without uncertainty guard | 总是接受最高 rank | 安全门是否降低 harmful changes |
| Full ETSF | 五成员、六头全局门、五头 root uncertainty、paired selector | 完整方法是否提高成功率 |

跨本体报告不能只给 pooled 指标。至少应分别给 Source 本体、Piper 目标本体、LOBO 未见本体，并报告：事件 macro-F1、success AUC/NLL/Brier、duration MAE/NLL/coverage、object L2/NLL、recovery AP、uncertainty AURC，以及最关键的 paired task-success gain、95% LCB、changed coverage 与 harmful-rate 95% UCB。

还应把“privileged simulator observer”和“非特权视觉/proprio observer”分开报告。前者可以作为 world-model/ranker 上界，不能冒充真实部署结果；后者才回答在线事件状态是否能够跨本体观测。

## 15. 当前实现与训练状态（2026-08-28 21:31 CST）

### 15.1 远端 4090 链路

- 服务器：`user@100.115.128.14`；GPU UUID `GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`，RTX 4090 D。
- 用户原有官方 OpenVLA 全量评测仍在运行。最近一次只读检查为 `402/500`、`231 successes`，GPU 利用率 98%。这不是 ETSF 训练或 ETSF 成功率结果。
- r12 Source watcher PID `2004304`，状态 `waiting_for_external_suite_parent_exit_before_rtx4090`。native core 初始化已完成，但五成员 Source 参数更新尚未开始。static plan SHA 为 `10ed8ceb1eb2d5374225df247fe078b220414d4994f5d970af8a0c552fa4aac4`。
- r12 LOBO watcher PID `2005507`，状态 `waiting_for_authenticated_source63_terminal_receipt`。
- r12 Piper watcher PID `2006249`，状态 `waiting_for_piper_then_ur5_lobo_terminal_no_hdf5_access`。该 watcher 的 `max_episode_steps=4`，只是可行性 smoke，不是 development300、Formal190 或正式 evaluation400。
- 三个 watcher 都是服务器端脱离进程，关闭本机不会终止。旧 r7h/r8e/r9b watcher 已在验证新链等待状态后停止；用户原有 OpenVLA 进程未被停止或修改。

远端 r12 Source 代码根为：

```text
/home/user/etsf_smolvla_source63_training_code_r12_20260828
```

该目录只读冻结，`writable_count=0`。Source 输出根为：

```text
/home/user/etsf_smolvla_schema5_native_source_training_r12_20260828
```

### 15.2 已完成的实现验证

- 因果历史 Source/Piper 及相邻链路：146 项 CPU 回归通过；覆盖未来隔离、padding/mask、截断、root bit-exact、branch 隔离和旧合同拒绝。
- Formal190 selection-aware 外层 OOF、identity bridge 与 paired downstream：79 项 CPU 回归通过；bridge 独立复算每折参数、190 组唯一覆盖、固定 bootstrap draws、LCB/UCB 和泄漏字段。
- evaluation400 v4 独立审计基础层：10 项 CPU 回归通过；覆盖一 ULP、bool 伪数、registry 篡改、recovery 重放/断链、过早解密、AAD/密钥/密文篡改和 pair-cluster 指标。
- r12 Piper detached watcher 定向回归：39 项通过。
- 最终全仓库 CPU 回归：1080 项全部通过；唯一 warning 是本机禁用 GPU 时 PyTorch 的 CUDA driver 探测，不涉及远端 4090 训练。

这些是代码与合成协议验证，不是模型预测精度或任务成功率。

### 15.3 尚未完成、因此不能声称的结论

- Piper development300 虽已预注册为 80 train / 30 internal validation / 190 formal，但 `collection_authorized=false`，当前没有目标监督采集。
- Formal190 真实 selector artifact 尚未生成；evaluation400 的 execution inventory 尚未批准，也没有执行。
- evaluation400 v4 审计基础层尚未接入 runner/executor；当前 v3 runner 仍是 privileged simulator root 路径。
- 现有证据不能证明跨本体任务成功率已经提高，也不能证明六头在真实 Piper 上达到目标精度。正式结论必须等待 Source → LOBO → Piper adapter → Formal190 → 完全未见 evaluation400 的顺序证据。

## 16. 主要代码入口

| 功能 | 文件 |
|---|---|
| 共享事件世界模型 | `scripts/openvla_etsf_event_world_model.py` |
| Source 同根反事实训练 | `scripts/train_openvla_etsf_counterfactual.py` |
| Piper 低容量跨本体 adapter | `scripts/train_smolvla_piper_schema6_embodiment_adapter.py` |
| 五成员六头校准与 Formal190 root OOF | `scripts/calibrate_smolvla_piper_adapter_ensemble.py` |
| evaluation400 identity bridge | `scripts/freeze_smolvla_piper_evaluation400_paired_identity_bridge_v2.py` |
| evaluation400 v3 root selector | `scripts/select_smolvla_piper_evaluation400_root_candidate_v3.py` |
| evaluation400 v3 condition runner | `scripts/run_smolvla_piper_evaluation400_condition_v3.py` |
| evaluation400 v3 sealed result evaluator | `scripts/evaluate_smolvla_piper_evaluation400_results_v3.py` |
| evaluation400 v4 独立审计基础层 | `scripts/smolvla_piper_evaluation400_audit_contract_v1.py` |

## 17. 当前实现身份

以下 SHA256 是本文对应的 r12 本地实现身份；任一文件变化后，旧 selector/runtime authority 都应失效并重新冻结：

| 组件 | SHA256 |
|---|---|
| Source causal-history trainer | `c2afbc6c1cbde7684caa40216f2a860547ee77810ef47b5dfdbecbec46f009a4` |
| Piper embodiment adapter trainer | `882994c7792180a24f47850a0441df4a3558507d661fd13d0273363cb97d7909` |
| shared deployment uncertainty | `9364a799a61cf953b307175d2100eaa8add496d501aaa264d722d4fdc9fd1b72` |
| Formal190 ensemble evaluator | `ca2c93754d21d5edb9ce9f346a3733c86230fe805e3cf9ef164fa2f7e176a07d` |
| six-head calibrator/root outer OOF | `59bbec96785b246302cbfd13d00d5e753aa11a8bb64c652ae54dc74a321faf15` |
| post-collection v3 | `619d434280afe19317dcb29400d1339728775cbe9565a255ea2617c3dc30cd85` |
| paired identity bridge v2 | `e0eb572595f2932713eb2f1b17191b56b3629e09a19cf5ec5ab7f701fc1c88b2` |
| paired success protocol v3 | `e4f9676b97190c9b82109345537d2f02717165b52ddcd8cbb04333055c49b5b9` |
| evaluation400 root selector v3 | `9665fe4f218d91c7323ac5253d1a2de3968e4fbe0f277282a40701961e63d0aa` |
| condition runner v3 | `f0752576d6cb2c1651bce8a837c6fdbff274811bf651a25d05ceb52842d8294c` |
| external executor v3 | `ab2e1ad1f0b8aea766f94a4abbb1cd94be49b775daf05a10146324addf8679ed` |
| sealed result evaluator v3 | `68a3a95bc10fb98f4f23692eff90e8144bf2d0af98574807df9ce9264d3f7a31` |
| evaluation400 v4 audit foundation | `e5841fa6086b8636e802b97e60618c427a24e2644e91e279721a0649166a8e10` |
| Piper r12 autonomous watcher | `d6ea99f26a10dfae624d1bd7a588a3ad2960c94e3be557c5150879d0ab0dff48` |
| 200-step full-horizon runtime config | `ed03f316ab74402def8311fe71e1b3d8dd9e10d96b874f764644de288c680b68` |
