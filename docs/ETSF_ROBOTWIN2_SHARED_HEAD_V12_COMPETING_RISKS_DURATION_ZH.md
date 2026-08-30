# ETSF RoboTwin2 共享事件头 v12：下一事件—持续时间竞争风险模型

> 状态：代码已实现，共享头、离线评估和插件协议的 152 项定向回归已通过；
> 尚未在五本体真实监督上完成重训，因此本文只说明模型与验证协议，
> 不声称已经提高跨本体成功率。

## 1. 为什么需要 v12

共享头的目标输入与输出是：

```text
当前规范事件状态 + 规范本体动作效果 + 事件年龄/剩余预算
    -> 下一事件分布
    -> 到事件边界的持续时间分布
    -> 成功、失败与条件恢复
    -> 对象状态变化
    -> 可分解的不确定性
```

v11 已经具备规范 27-D 状态、14-D SE(3) 动作效果、post/next event、右删失
duration、有限时域 terminal event/success/recovery、对象 Student-t 分布、五成员
ensemble 和单调后果 utility。审计发现 duration 仍有一个实质性错位：五个 duration
分量按 `current_event_id` 硬选一个分量。这样表达的是

```text
p(D | current event)
```

而不是任务真正需要的

```text
p(next event, D | current state, action effect)
```

同一根状态与动作可能到达不同事件边界，且各目的事件具有不同时间尺度。硬选当前事件
会把这些模式压入一个对数正态分布；在不同机器人速度、接触动力学和失败模式混合时，
这会直接损害时长预测及其不确定性。

## 2. v12 联合分布

模型保留五路 clock 输出，但重新定义为：

```text
next_event_logits[e]             = p(e_next=e | state, action effect)
duration_component_log_mean[e]   = μ_e
duration_component_log_scale[e]  = log σ_e
log(1 + D) | e_next=e ~ Normal(μ_e, σ_e²)
```

因此联合分布为：

```text
p(e_next=e, D) = p(e_next=e) p(D | e_next=e)
```

这一分解仍然完全跨本体共享：没有新增 body embedding、机器人 ID 条件头或目标本体
统计量。速度差异通过规范动作效果、`dt`、事件年龄和隔离 clock 表达；LOBO 留出本体
的 payload、标签和归一化统计仍不进入训练或选择。

## 3. 完整监督与右删失

### 3.1 已观测边界

当 episode 内实际观察到下一事件 `e*` 和持续时间 `D` 时，训练项是：

```text
CE(e*, next_event_logits)
+ NLL(log1p(D); μ_e*, σ_e*)
```

duration 不再读取 `current_event_id`。数据加载器强制要求
`duration_observed == 1 -> next_event_mask == 1`，避免使用未知目的事件训练条件分量。

### 3.2 右删失边界

当有限时域结束但下一事件尚未到达时，目的事件未知。v12 使用竞争风险混合生存概率：

```text
S(D_c) = sum_e p(e_next=e) S_e(D_c)
L_censored = -log S(D_c)
```

这使一个删失样本同时约束所有可能目的事件的时间尾部，并且把梯度传给 next-event
概率和 duration 分量。它不会再把“尚未到达”伪装成当前事件时长，也不会凭空生成
下一事件标签。

## 4. 不确定性

对单成员，v12 在 `Y=log1p(D)` 空间计算下一事件混合的一、二阶矩：

```text
E[Y]   = sum_e p_e μ_e
Var[Y] = sum_e p_e (σ_e² + (μ_e - E[Y])²)
```

其中第一项是目的事件内 aleatoric uncertainty，第二项是目的事件之间的模态不确定性。
输出字段为：

- `duration_mixture_probability`；
- `duration_log1p_mixture_mean`；
- `duration_log1p_mixture_std`；
- `duration_component_log_mean/log_scale`。

五成员间的预测差异继续提供 epistemic uncertainty。离线 LOBO 评估对 observed duration
按 ensemble 联合分布条件化，对 censored duration 先在每成员内对目的事件求和，再在五成员
间做 mixture，而不是对一个矩匹配单峰计算伪精确 NLL。

模型还显式输出 `success_probability`、`failure_probability`、
`conditional_recovery_probability`、`joint_recovery_probability`，以及 post/next/terminal
event、success、conditional recovery 和 joint recovery 的单成员 aleatoric entropy。它们不替代
五成员 epistemic disagreement，也不表示已经完成温度校准；最终 calibration 参数仍只能在
LOBO 外层训练折内部拟合并在留出本体上验证。

## 5. 与成功、恢复和对象效果的关系

v12 没有把所有任务压成一个自由 critic 标量：

- `post_event` 和 `next_event` 保持分类 proper loss；
- `success` 仍严格等于有限时域 terminal event 的 `eK` 概率；
- `recovery` 仍是发生 operational regression 后的条件概率；
- 对象变化仍是 `p(post_event) p(delta_object | post_event)` 的 Student-t(3) 混合；
- terminal goal progress 仍按 terminal event 条件化；
- candidate utility 只读取 detach 后的有界后果特征。

持续时间现在与 next event 形成和对象效果相同的“离散事件 + 条件连续量”结构，使事件
建模在语义上闭合，但不会让排序损失篡改 proper prediction heads。

## 6. 兼容性

模型族升级为：

```text
terminal_consequence_utility_shared_event_head_v12
```

参数形状没有变化，仍使用原来的五路 `duration_mean/duration_scale` 线性层。为了兼容现有
paired runner、postformal candidate pool 和校准器，保留：

- `duration_log_mean/log_scale [B,5]`；
- `duration_selected_log_mean/log_scale [B]`。

后两个字段不再表示硬选当前事件分量，而是 `log1p(D)` 混合的一、二阶矩等价 Gaussian。
因此现有标量接口无需迁移即可得到包含“事件内 + 事件间”的时长不确定性；要求精确密度
的训练/离线评估必须读取 component 和 mixture probability 字段。

旧 v11 checkpoint 的参数张量可以装入 v12 类，但它没有接受 v12 竞争风险目标训练，不能
改名冒充 v12 结果。正式五折 LOBO 必须从 v12 训练入口重新训练五成员。

## 7. 验证要求

代码回归覆盖：

1. 混合方差同时包含 within-event 和 between-event 两项；
2. observed duration 由真实下一事件分量解释，而不是当前事件；
3. censored NLL 等于概率加权的竞争风险生存函数；
4. censored loss 对 next-event logits、duration mean 和 scale 都有非零梯度；
5. 旧标量输出保持 `[B]`，组件输出保持 `[B,5]`；
6. 完整共享头训练/排序/消融合同回归不退化。

真实效果仍必须由五折 LOBO 回答，至少报告：

- held-out body next-event macro-F1/NLL/ECE；
- observed duration MAE 和条件 mixture NLL；
- censored competing-risks NLL；
- duration uncertainty-error AURC；
- success Brier/NLL、recovery、对象 Student-t NLL；
- 最终 frozen actor + shared head 相对 actor 的配对 `Delta SR` 与 `Delta Stage`。

只有五个留出本体均有完整预测证据，且正式配对成功率置信区间支持改善，才能声称 v12
实现了可迁移、可改善的事件世界模型；离线 loss 下降本身不等于跨本体任务成功率提升。
