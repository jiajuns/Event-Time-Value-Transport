# RoboTwin2 matched WCM-style future-latent 基线 v1

## 定位与边界

这是一个为共享事件头设计的**同数据、同动作协议、近似同容量、同训练预算**的
WCM-style 对照基线。它借鉴 WCM/LeWM 的“历史条件价值预测 + 动作条件未来潜变量”
思路，但不是官方 WCM 网络、官方 checkpoint 或官方权重的复现。参考实现与论文：

- WCM 官方仓库：<https://github.com/sylvestf/WCM>
- WCM 论文：<https://arxiv.org/abs/2607.29613>

当前 branch 数据没有未来图像、未来视觉 token 或完整 `post_state[27]`。因此本基线
不会伪造视觉 future latent，也不声称是像素世界模型；监督目标严格定义为
**有限时域 canonical terminal consequence/effect latent**。如果之后采到根状态对应的
完整终态观测，可以另开版本加入 visual/post-state target，不能在本版本里暗中替换。

## 公平匹配合同

训练器直接复用共享头 v13 已冻结的数据权威与协议：

- primary binding 和可选 proper-world supplement binding；
- 五体 LOBO：held-out body 只用于清单分组，其 manifest payload/NPZ 在训练、
  归一化和 checkpoint 选择阶段保持 zero-open；
- label-blind source train/validation split；
- 27-D canonical state、14-D canonical action、相同 EE16 世界坐标 frame；
- 相同冻结 actor execution protocol；
- primary source train-only state/action normalization，supplement、validation、held-out
  均不参与拟合；
- primary 的 body/condition/current-event 因果均衡、五成员 outcome-preserving
  group bootstrap；
- supplement 训练与 source validation 选择权重固定为 `0.25`；
- 五成员共同 source-validation step 选点，不允许每个成员各挑不同 step。

默认预算与 v13 一致：每成员 3000 step、每 100 step 验证、batch 64、AdamW
学习率 `3e-4`、weight decay `1e-4`、五成员。默认模型有 221,558 个可训练参数，
共享头 v13 参考值是 223,287，比例为 0.9923，处于预注册的 0.95–1.05 匹配区间。
body/condition 不作为可训练输入；所有 source body 在 dataset 中共用 `body_id=0`。

## 模型

推理输入只包含当前可因果获得的信息：

```text
state27 + event_age + remaining_action_budget + dt
                         │
                         ├── context encoder ─────────────┐
candidate action14[H] ───┴── token MLP + masked GRU ─────┤
                                                         ↓
                                              action-conditioned dynamics
                                                         ↓
                                           predicted future latent (96-D)
                                                         ↓
                          terminal event / success / value / object effect
```

branch 执行完成后才构造 14-D target：

```text
5-D terminal event one-hot
+ success
+ terminal stage progress
+ bounded terminal goal progress
+ 6-D bounded object pose effect
```

target encoder 只在训练时存在。runtime scorer 如果收到 success、terminal event、
stage、goal、object effect 或它们的 mask，会直接 fail-closed。

## 损失与防坍塌

总损失包含：

1. predicted future latent 到 detached target latent 的 MSE；
2. success Bernoulli NLL；
3. stage/goal diagonal-Gaussian NLL；
4. terminal-event categorical NLL；
5. object-effect diagonal-Gaussian NLL；
6. target encoder 的 success/value proper decoder 辅助损失；
7. batch-level characteristic-function SIGReg；
8. batch-level variance/covariance 防坍塌项。

默认权重依次为 `1, 1, 1, 0.5, 0.5, 0.25, 0.01, 0.01`。SIGReg 使用随机单位
投影上的经验特征函数与标准高斯特征函数差异；variance/covariance 项抑制低方差维度
和非对角协方差。两项都服从 member bootstrap/sample weight：零权重 branch 的真实
outcome 不会通过 target encoder 的正则项泄漏给该 member。只有 success、terminal
event、goal、object effect 四类监督都有效的 row 才进入 latent MSE、target SIGReg/
variance-covariance 和 target-decoder 辅助项；不完整 row 仍可按各自 mask 训练相应的
predicted proper head，masked placeholder 不能进入 future target latent。

checkpoint 选择只使用 source validation 的 proper prediction loss：success、value、
terminal event、object effect；latent MSE、SIGReg、variance/covariance 只作诊断，不能
利用 held-out 标签选点。

## 五成员 checkpoint 与 N4/N8 排序

每个 fold 输出：

```text
member_00_seed_..._best.pt
...
member_04_seed_..._best.pt
training_summary.json
preflight_receipt.json
```

生产加载器要求五个成员满足同 held-out body、同 source bodies、同 common step、
同 normalization、同 primary/supplement binding、同 actor protocol，且 member 顺序必须
为 0–4、seed 必须互异。checkpoint 同时绑定 canonical schemas、frame contract、event
spec、参数量和 zero-heldout 收据；任何字段或协议漂移都会拒绝加载。

单成员候选分数为：

```text
p(success) + 0.25 * clamp(predicted_stage, 0, 1)
           + 0.05 * tanh(predicted_bounded_goal_progress)
```

五成员汇总为 `mean(score) - 0.25 * population_std(score)`。正式 scorer 只接受同一
logical root 的严格 `candidate_index=0..N-1`，且 `N∈{4,8}`；分数相同时由
`torch.argmax` 固定选择最低 candidate index。

该排序器只是 critic baseline。跨本体主结论仍必须来自 held-out body 上冻结 actor 的
同 seed 配对 task success / stage progress；critic 是 LOBO transfer，不能把 actor 本身
误写成 zero-shot，也不能用 AUC 或 source-validation proper loss冒充迁移成功率。

## 使用方式

先对单个 held-out body 做只读 preflight：

```bash
PYTHONPATH=scripts python scripts/train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py \
  --mode preflight \
  --binding /abs/path/primary_binding.json \
  --binding-sha256 <primary_sha256> \
  --supplement-binding /abs/path/supplement_binding.json \
  --supplement-binding-sha256 <supplement_sha256> \
  --held-out-body ur5
```

在授权 RTX 4090 上训练该 fold：

```bash
PYTHONPATH=scripts python scripts/train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py \
  --mode train-fold \
  --binding /abs/path/primary_binding.json \
  --binding-sha256 <primary_sha256> \
  --supplement-binding /abs/path/supplement_binding.json \
  --supplement-binding-sha256 <supplement_sha256> \
  --held-out-body ur5 \
  --output /new/path/wcm_lobo_ur5
```

分别以 `aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5` 为 held-out body 执行，
得到五个独立 LOBO fold。此实现没有远程自动启动或续跑逻辑，避免和现有 RAC/nested
runner 混合；训练前应由上层实验编排明确绑定输入 SHA 与全新的输出目录。

本地最小合同测试：

```bash
PYTHONPATH=scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_robotwin2_wcm_future_latent_baseline_v1.py
```
