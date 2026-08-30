# ETSF RoboTwin2 跨本体共享事件头 v10 设计与实验协议

## 1. 当前结论

v10 是冻结策略外部的、动作条件化的事件后果模型。它不生成动作，而是在同一状态下对冻结 SmolVLA 产生的多个真实候选动作排序，再执行最高分候选。模型参数在五种机器人之间共享，训练和模型选择严格使用四个源本体；留出本体的监督、补充数据与结果不会进入训练、归一化、校准或 checkpoint 选择。

截至 2026-08-31，正式五本体分支采集仍在远程 RTX 4090 上进行，因此尚不能声称跨本体成功率已经提高。当前已有的是完整实现、数据合同和 CPU 回归验证；最终效果只由留出本体闭环配对实验的 `Δ成功率` 与 `Δ阶段进度` 决定。

## 2. 要解决的问题

旧数据根事件主要是 `e0/e12`，旧补充数据只有 `e3/e4`，缺少任务中最关键的 `e12→e3` 边界监督；同时，终止事件虽然是有序阶段，旧 proper loss 只使用分类交叉熵；旧正式 N4 的四个 flow noise 又只有两个独立随机基向量。结果是模型可以在离线指标上学习后果，却未必获得足够的边界信息和候选覆盖来改善闭环成功率。

v10 直接针对这三点修改：

1. 补充数据同时采集 `e12/e3/e4` 三个专家根状态，并从每个根只运行冻结 actor 的真实候选和后续轨迹。
2. 终止事件同时使用分类交叉熵和严格适当的有序 Ranked Probability Score（RPS）。
3. 补充集的真实组内候选差异以低权重直接监督单调效用排序头；它不反向影响事件后果特征，也不参与验证或选模。
4. 最终评估使用共享 raw16 的 actor/N4/N8 三臂设计，N4 是 N8 的严格前缀。

## 3. 事件与输入表示

事件按物理进度固定为：

```text
e0  未移动
e12 已移动或抬起
e3  已到目标附近
e4  在目标附近稳定
eK  模拟器判定成功
```

共享头的单行输入是：

```text
canonical state s_t
+ canonical candidate effect u_t
+ current event e_t
+ event age
+ remaining action horizon
```

`canonical state` 和 `canonical candidate effect` 都是任务/物理坐标中的统一表示，不含机器人 ID 专属的动作槽位。候选效果取将执行的前五个动作，并使用相对末端位姿、旋转与夹爪变化表示。因此，不同本体只需要各自的策略和 executor，事件头本身不需要本体专属参数。

## 4. 模型输出

模型族名称为：

```text
terminal_consequence_utility_shared_event_head_v10
```

对每个候选动作，五成员深度集成共同预测：

- 当前动作段后的事件分布 `p(e_post)`；
- 下一事件分布 `p(e_next)`；
- 事件持续时间的 log-normal 分布；
- 终局事件分布与成功概率；
- 失败恢复概率；
- 对象 SE(3) 状态变化的 Student-t 分布；
- 终局目标进度的 Student-t 分布；
- aleatoric 与 ensemble epistemic 不确定性；
- 由上述后果构成的有界、单调候选效用。

在线分数采用五成员风险调整聚合：

```text
score = ensemble_mean(utility) - 0.25 * ensemble_population_std(utility)
```

actor 不更新，事件头不向策略回传梯度。

## 5. v10 训练目标

### 5.1 正式源本体数据

正式数据继续训练事件分类、持续时间、成功/恢复、对象效果、终局事件和目标进度等 proper world-model 目标，并使用同根四候选的真实成功差异或全失败阶段差异训练候选排序。

终局事件新增有序 RPS：

```text
L_terminal_event = 0.5 * CE(p, y) + 0.25 * RPS(p, y)
```

CE 保留完整类别概率的识别能力；RPS 让把 `e4` 预测成 `e3` 的代价小于预测成 `e0`，但仍是对完整概率分布严格适当的评分规则，不需要伪造连续标签或成功样本。

### 5.2 e12/e3/e4 补充数据 v2

设计规模为：

```text
5 bodies × 2 conditions × 5 horizons × 3 event roots
= 150 decisions
= 600 real actor branches
```

每个 body/condition/horizon 槽位先冻结一个有序 reserve seed roster。公开 RoboTwin 专家只用于走到首个 `e12/e3/e4` 物理状态并保存可恢复 snapshot；选中三元根以后专家立即退出，所有候选和后续结果均来自与正式实验相同的冻结 SmolVLA。

补充损失为：

```text
L_supplement = 0.25 * L_proper_world + 0.25 * L_detached_utility_rank
```

排序项只接收 detached 后果特征，只更新 candidate utility 参数。补充行明确不用于：

- state/action normalization；
- source validation；
- checkpoint selection；
- calibration；
- semantic comparative world-model loss；
- 留出本体训练。

每个 ensemble member 对真实逻辑组独立做 Poisson bootstrap，不做类别平衡，也不生成合成成功标签。

## 6. 严格跨本体协议

每一折留出一个机器人：

```text
train/normalize/bootstrap: 其余四个本体
validation/model selection: 其余四个本体的 source-only split
held-out body payload access before frozen checkpoint: 0
held-out execution: checkpoint 完全冻结后才开始
```

五折分别留出 `aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5`。最终宏平均对五个本体等权，而不是按 rollout 数量加权。

这种设计证明的是同一 canonical 事件后果模型是否能迁移到从未参与训练的机器人；它不依赖共享策略权重，也不把训练过该任务的 actor 本身误称为 critic 跨本体证据。

## 7. 可插拔接口

事件头只要求候选生成器提供：

```text
state observations
current native EE state
N candidate action chunks
event/horizon context
```

运行时适配器把 native action chunk 转成统一 candidate effect batch，事件头返回每个候选的风险调整分数和不确定性，原 executor 执行选中候选。因此它可接 SmolVLA、OpenVLA/OFT、diffusion policy 或其他能够输出多个候选的策略。迁移到新任务时仍需要定义任务事件与对象 canonicalizer；“跨本体”不等于“不经任务适配跨任意任务”。

## 8. 候选池与强因果评估

最终 enhanced 评估不再把两个独立 N4/N8 实验直接相减，而是每个初始条件运行三个 fresh rollout：

```text
actor baseline: raw candidate 0
N4: shared raw16 的 FPS 顺序前 4 个
N8: 同一 FPS 顺序前 8 个
```

raw16 使用两个语义等价指令条件与八个配对独立 flow noise。candidate0 保持旧运行时短指令和旧 noise0，确保 actor baseline 身份不变。FPS 只读取前五步 canonical action effect，不读取事件、结果或 critic 分数。

三臂在同一 seed 下重新完整 reset，执行顺序按 seed 轮换。query0 的 raw16、N4/N8 索引和 reset 身份在任何一臂执行前冻结。这样可以分别报告：

- `N4 - actor`：共享头 best-of-4 的闭环收益；
- `N8 - actor`：共享头 best-of-8 的闭环收益；
- `N8 - N4`：严格扩大可选候选池的因果收益。

## 9. 最终报告指标

主指标不是 AUC、MAE 或 Brier，而是留出本体闭环迁移效果：

```text
ΔSR = SR(actor + shared head) - SR(actor)
ΔStage = Stage(actor + shared head) - Stage(actor)
```

报告内容包括：

- 五本体等权 macro `ΔSR` 和 `ΔStage`；
- 每本体、每 clean/randomized 条件结果；
- 配对 bootstrap 95% 置信区间；
- 成功二值差异的 exact McNemar 检验；
- actor→N4、actor→N8、N4→N8 三组比较；
- 事件预测、持续时间、Brier/RPS 等诊断指标。

只有完整 1000 个初始条件、3000 个三臂 rollout 的 completion receipt 存在后，才能对成功率效果下结论。离线 critic 指标只作为机制证据。

## 10. 代码入口

- v10 trainer：`scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- e12/e3/e4 补充采集：`scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- 补充 binding：`scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
- postformal 候选池：`scripts/run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- shared-raw16 三臂评估：`scripts/run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py`
- detached upgrade watcher：`scripts/watch_robotwin2_postformal_shared_head_upgrade_v1.py`

## 11. 当前限制

冻结 SmolVLA actor 只训练约 1.06 epoch，当前正式分支中的成功上限很低。v10 能改善候选后果建模和选择，但不能从候选池中创造 actor 从未提出的可成功动作。若完整 raw16 的离线/闭环 oracle 仍没有成功 headroom，下一步应续训 actor 或增加真实多样候选生成，而不是继续堆叠 critic 门控。
