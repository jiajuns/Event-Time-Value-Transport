# ETSF RoboTwin2 共享事件头 v13：源域因果分层平衡

> 状态：模型与五折 LOBO 训练合同已经实现；本地 trainer、LOBO watcher 和自动续跑相关
> 回归通过。远程完整训练尚未结束，因此本文说明 v13 的改动、监督与验证方法，不把代码通过
> 测试写成跨本体成功率已经提高。

## 1. v13 解决的实际问题

v12 已经建立动作条件的结构化事件世界模型：

```text
当前事件状态 e_t + 规范动作效果 u_t + 事件年龄/剩余预算
    -> p(post event)
    -> p(next event, duration)
    -> p(terminal event / success / failure / recovery)
    -> p(object state change)
    -> uncertainty + bounded candidate utility
```

但 primary 数据按 `body × condition × query × seed` 采集均衡，不代表实际到达的
`current_event` 均衡。早期事件更常见，容易本体也更容易进入某些事件；如果所有 branch 直接按
经验频率优化，多任务 proper loss 会主要服务高频早期事件。增大网络或继续加选择门不能修复这种
监督质量问题。

v13 保留 v12 的参数、forward 和候选排序接口，只改变 source-train proper-likelihood 的抽样质量：
让每个源本体、条件和当前事件在训练梯度中获得更合理的质量，同时避免完全逆频率的高方差。

## 2. 因果分层权重

一个决策组包含同一根状态的四个候选。对 source-train 决策组 `g`，定义动作执行前可见分层：

```text
s(g) = (body, condition, current_event_id)
n_s  = 该分层中的完整决策组数量
w_raw(g) = n_s^(-1/2)
w(g) = w_raw(g) / mean_all_source_train_groups(w_raw)
```

最终 proper loss 权重为：

```text
w_final(member, g) = group_poisson_bootstrap(member, g) * w(g)
```

选择平方根逆频率而不是完全逆频率的原因是：当两个分层数量比为 `4:1` 时，经验训练的有效质量比
是 `4:1`，v13 将其压到 `2:1`；完全逆频率会直接压到 `1:1`，但会让极少事件产生过大的梯度
方差。所有权重在四候选内完全相同，不会根据某个候选结果改变训练质量。

## 3. 不读取结果标签

权重计算只允许读取：

- source body；
- clean/randomized condition；
- 动作执行前的 `current_event_id`；
- logical decision group 与四个 candidate index。

禁止读取 success、failure、recovery、post/next event、duration、object effect、terminal outcome、
candidate utility 或任何 held-out body payload。测试会把全部 outcome 字段替换成不可解释对象，并
要求权重与审计 SHA 完全不变。

supplement 数据和 source validation 保持经验分布，不做相同重权：supplement 数量小且承担专门的
事件/排序补充作用；validation 必须估计自然分布上的 proper score，不能为了让 v13 看起来更好而
改变评估分布。

## 4. 跨本体边界

v13 没有添加 body embedding、目标机器人 adapter、本体私有 head 或目标本体统计量。每个外层
LOBO fold 只用另外四个 source bodies 计算权重；held-out body 的 manifest、NPZ、标签、频数和
归一化统计在训练与 checkpoint 选择期间均为 zero-open。

实现上，真实 NPZ materializer 必须把 manifest-visible `condition` 作为非张量元数据附到每个
source row；`TransitionDataset` 在送入模型前显式剥离该字段。这样因果分层器可以使用
`(body, condition, current_event)`，但网络本身不会把 condition 当作可训练输入。2026-08-31 的入口
审计补齐了该字段并新增真实 materialization 回归，避免单元测试只用手工 row 而遗漏生产入口。

因此 v13 改善的是“源域多本体监督怎样训练一个共享事件头”，而不是偷偷用目标本体校准。最终
仍只能主张 critic/value-head 的 LOBO 跨本体迁移；当前 actor 在五种机器人数据上训练过，不能把
结果表述为 actor zero-shot。

## 5. 兼容性和消融

模型族：

```text
terminal_consequence_utility_shared_event_head_v13
```

v13 不改变任何参数张量或推理字段，所以 state dict、五成员 ensemble、risk-adjusted utility、
N=1/4/8 runner 和外部插件接口保持兼容。但旧 v12 checkpoint 没有按 v13 objective 训练，不能只改
metadata 冒充 v13。

训练入口：

```text
--proper-balance-mode causal_body_condition_event_sqrt  # v13 默认/正式
--proper-balance-mode empirical                         # 精确恢复 v12 经验分布对照
```

这一消融轴与 `success_only / no_time_duration / no_object_effect / full` 正交。正式对比必须使用相同
source split、ensemble seeds、3000 steps、checkpoint 选择、候选池和 held-out rollout seeds。

## 6. 必须落盘的审计

每个 checkpoint 与 training summary 记录：

- balance mode 与是否启用；
- source body roster、决策组数和候选行数；
- 因果分层数、每层最小/最大决策数；
- 经验质量比与重权后的有效质量比；
- 最小、最大、平均 group weight；
- `(group, body, condition, current_event, weight)` 的 canonical SHA；
- `outcome_label_fields_read=[]` 与 `heldout_rows_used=0`。

五折 watcher 显式传入 v13 mode，并拒绝 model family、balance mode 或 heldout audit 不一致的
checkpoint；最终 five-fold aggregate 同样记录模型族和固定 balance mode。

## 7. 效果验证

v13 预计改善的是低频事件与困难本体的 next-event、duration、terminal consequence 和 candidate
ranking，而不是保证所有指标自动变好。完整验证分两层：

1. 机制层：逐 `body × condition × current_event` 报告 macro proper loss、next-event F1/NLL、
   duration MAE/NLL、success Brier/NLL、recovery、object NLL 与 uncertainty AURC；对比 v13 与
   empirical control，不能只报 pooled mean。
2. 主张层：冻结同一 actor，在五个 held-out body 上以同 seed 完成 N=1/4/8 配对闭环，报告每个
   body、condition 和等权宏平均的 `Delta SR`、`Delta Stage`、requested-seed cluster CI 与 exact
   McNemar。只有完整结果支持改善，才能写“跨本体共享事件头提高任务成功率”。

若 v13 离线 proper score 改善但闭环 `Delta SR` 没有改善，应检查 candidate oracle headroom 和
ranking regret，而不是继续增加门控；若 oracle headroom 本身接近零，瓶颈在 actor 候选覆盖而不在
共享头。

## 8. 与最新相关工作的区别

VLAC 和 ProgressVLA 已经证明 progress critic、动作条件未来预测和 world-model guidance 有效；
VLA-ATTC 已经用 Relative Action Critic 做 best-of-N；WCM/WVM 已经把未来 latent 预测加入 value
estimation。因此创新不能写成“首次用世界模型或 critic 选动作”。

当前最可验证的定位是：**在冻结异构 actor 外部，用规范物理动作效果预测事件、竞争风险持续时间、
恢复和对象后果；以 source-only 因果分层训练保持跨本体共享，并在严格 held-out-body best-of-N 中
同时用 proper prediction 与配对任务成功率验证。** 标量 RAC 和非结构化 future-latent critic 应按
相同数据、容量、候选与 rollout 预算重建为正式基线。
