# Event-Time Value Transport

Event-Time Value Transport（ETSF）研究跨机器人本体的价值函数迁移：当机器人关节结构、控制器或执行速度发生变化时，能否避免在目标本体上重新训练 critic，仅用少量 rollout 估计事件可达性和局部时间变化，将源域价值传输到目标本体。

## 当前状态

| 阶段 | 状态 | 结论 |
|---|---|---|
| Stage 0 | 已完成 | VOC 会忽略保持秩序的时间重参数化；折扣饱和会破坏价值秩 |
| Stage 1 | 已完成 | 事件结构大体可共享，`ρ_b` 是主要迁移因素，旧事件链和 AUC 口径存在问题 |
| Stage 2 G0–G2 | 已完成 | 事件链修复、后验认证、共享液态头及模块边界测试通过 |
| Stage 2 G3 | **未通过** | T4 改善 Bellman 一致性和局部时长方向，但没有同时保持 AUC 与优势符号 |

当前结果不能表述为“已经完成零样本跨本体 critic 迁移”。代码按预注册规则在 G3 止损，没有继续 T5、target-from-scratch 和完整 V3。

## 实际问题与方法

源机器人上训练的价值函数通常绑定源机器人的控制时钟。两个机器人即使完成相同事件序列，也可能在不同事件段具有不同速度，使折扣、价值排序和优势符号在目标机器人上失真。

本项目把跨本体价值迁移分解为：

- `ρ_b`：目标本体沿规范事件链继续推进的后验可达率。
- `β_b`：目标本体唯一的私有时间标量，只调制共享液态头的局部时间常数。
- `Ψ^liq_θ`：跨本体硬共享的液态事件后继价值头。

核心形式为：

```text
V_b = Transport(Ψ^liq_θ; ρ_b, β_b)
log τ_b(h, x) = a_θ(h, x) + β_b g_θ(h, x)
```

目标适配只估计 `O(K+1)` 个统计量，不更新共享 critic，也不执行目标域 TD。价值输出采用 HL-Gauss 分类和随局部折扣解析变化的动态 support。

## 仓库结构

```text
configs/
  stage0_gate1.yml
docs/
  ETSF_agent_runbook.md
  ETSF_stage1_overnight.md
  ETSF_stage2_liquid_transport.md
scripts/
  run_stage0_experiment.py
  run_gamma_sensitivity.py
  run_stage1.py
  run_stage2.py
  ...
```

- [总运行手册](docs/ETSF_agent_runbook.md)
- [Stage 1 执行文档](docs/ETSF_stage1_overnight.md)
- [Stage 2 液态价值传输文档](docs/ETSF_stage2_liquid_transport.md)
- [Stage 2 正式入口](scripts/run_stage2.py)

仓库不包含轨迹数据、图像特征、测试文件、实验结果、模型权重、日志或 checkpoint。独立测试文件和 probe 文件已通过 `.gitignore` 排除；必要的模块边界检查内嵌在正式 Stage 2 入口中。

## 4090 环境

已验证环境：

```text
GPU: NVIDIA GeForce RTX 4090 D, 24 GB
Python: 3.10
PyTorch: 2.10.0+cu128
Environment: /home/user/anaconda3/envs/ETSF_RoboTwin
```

主要依赖：

```text
torch numpy pandas scipy h5py scikit-learn matplotlib
```

Stage 2 内联了最小闭式低秩液态 cell，不依赖 `ncps`。这不是把 `β_b` 当作普通输入，而是将其限制在液态时间常数中。

## 数据目录

默认数据根目录为：

```text
/home/user/etsf_stage1
```

至少需要：

```text
/home/user/etsf_stage1/source_object_poses/
/home/user/etsf_stage1/target_rollouts/
/home/user/etsf_stage1/m2_fresh/
```

数据规模：

- 源本体：Aloha、ARX-X5，每任务各 50 条成功轨迹。
- 留出本体：Piper、UR5-WSG，每任务各 20 条未筛选 rollout。
- 共 6 个任务、840 条正式轨迹。

事件字母表和阈值只使用 Aloha、ARX-X5 标定。Piper、UR5-WSG 不参与事件选择。`e1/e2` 在源数据中几乎同时发生或顺序冲突时合并为无序接管事件 `e12`；语义上不可能触发的事件不会进入该任务规范链。

## 快速运行

以下命令可在已配置的 4090 服务器上直接执行。

### 1. 模块自检

```bash
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  scripts/run_stage2.py --stage self-test
```

期望输出：

```text
SELF_TEST_PASS=posterior,polarity,split
```

### 2. G0：冻结数据与事件口径

```bash
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  scripts/run_stage2.py \
  --stage g0 \
  --data-root /home/user/etsf_stage1 \
  --output-root /home/user/etsf_stage2
```

G0 会执行：

- 从源本体导出每任务规范事件链。
- 修复 `e1/e2` 退化和永不触发事件。
- 固定每任务前 15 条适配池、后 5 条测试集。
- 重算 M1、秩检验和 N=5/10/15 后验预测。
- 复现 Piper/adjust_bottle 的 anytime-valid 序贯认证。
- 审计控制时间基线的 AUC 极性。

### 3. 正式训练和 N=5 留出测试

```bash
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  scripts/run_stage2.py \
  --stage main \
  --steps 3000 \
  --adaptation-count 5 \
  --data-root /home/user/etsf_stage1 \
  --output-root /home/user/etsf_stage2
```

该入口训练并比较：

| 模型 | 时间参数 | 说明 |
|---|---|---|
| T1 | 全局 `λ_b` | MLP 全局时间基线 |
| T2 | 逐事件 `λ_b[j]` | 可表达事件级反向变化的强对照 |
| T3 | 全局液态 `β_b` | 共享液态头，全局时间调制 |
| T4 | `a_θ + β_b g_θ` | 低秩状态相关液态调制，主方法 |

共享头只使用物体相对状态、累计位移、归一化运动方向和任务/事件 token。机器人状态、本体 ID、关节角、机械臂图像和原始速度幅值不会输入共享头。

重复运行会覆盖 `--output-root` 下的同名结果，请在需要保留旧结果时使用新的输出目录。

## 输出

默认输出目录：

```text
/home/user/etsf_stage2/
```

主要文件：

```text
g0_summary.json
main_summary.json
stage2_report.md
event_spec.json
stage2_models.pt
results/
  P1_fixed_event_consistency.csv
  P1_rank_test.csv
  P1_n_sweep_corrected.csv
  P1_auc_budget_audit.csv
  rho_deployment_approval.csv
  V0_value_transport.csv
  V1_sign_reversal.csv
  V2_kappa_reproduction.csv
  T_ladder_comparison.csv
  beta_convergence.csv
  leak_check.csv
  raw_metrics.csv
```

这些输出位于运行服务器，不会自动写入 Git 仓库。

## 判定指标

- **Bellman MSE**：在同一事件边界口径下比较解码价值与事件 bootstrap 目标。
- **价值排序 AUC**：成功轨迹的初始价值是否高于失败轨迹；分数方向统一为“越大越接近成功”。
- **优势符号一致率**：逐边界比较 `sgn_ε(A)`，`|A|≤ε` 记为 0；不是价值曲线单调比例。
- **局部方向一致率**：模型能否判断目标本体在每个事件段相对参考本体更快、更慢或无显著变化。
- **成功率 MAE**：事件链完整 Beta 后验给出的成功概率与严格留出轨迹成功率之差。

目标域拟合后，代码逐元素检查共享头参数不变。泄漏检查还包括未来扰动、假本体 ID、动态 support、HL-Gauss 归一化以及 `β=0` 的线性本体 probe。

## 当前正式结果

随机种子固定为 `20260825`，目标适配为每任务前 5 条，测试为最后 5 条。

### G0

| 指标 | 结果 |
|---|---:|
| 结构偏序一致率 | 1.0000 |
| 干净格子秩-1 解释率 | 0.9341 |
| N=5 可行性预测 MAE | 0.3025 |
| 数据划分重叠 | 0 |

Piper/adjust_bottle 在第 12 条 fresh rollout 停止，拒绝部署后验为 `0.990311`，anytime `p=0.042922`。

### G1/G2

- 8 项边界与泄漏检查全部通过。
- `β=0` 线性本体 probe：accuracy `0.3083`，机会水平 `0.25`，`p=0.0874`，未检出显著本体泄漏。
- UR5-WSG 局部方向一致率：T3 `0.6429`，T4 `0.8571`。

### G3：真正留出的 `{Aloha, ARX-X5} → UR5-WSG`

| 模型 | Bellman MSE | AUC | 优势符号一致率 |
|---|---:|---:|---:|
| T2 | 0.010667 | 0.8622 | — |
| T3 | 0.016203 | 0.8622 | 0.6792 |
| T4 | **0.009997** | 0.8400 | 0.5660 |

T4 的 Bellman MSE 和局部方向优于 T3，但 AUC 低于 T2/T3，优势符号一致率也低于 T3。因此 G3 严格判定为未通过。

## 止损与研究边界

当前能够支持的结论是：

> 低秩液态调制能更好地拟合局部时钟，并改善事件边界 Bellman 一致性，但这些改善尚未可靠传递到价值排序和策略更新所依赖的优势符号。

当前不能支持的结论是：

> 一个目标标量已经足以完成零样本跨本体 critic 迁移。

共享头训练使用 Aloha 与 ARX-X5，所以 `aloha → ARX-X5` 只能作为源域时钟诊断，不能称为留出本体迁移。真正的 G3 目标是未参与共享头训练的 UR5-WSG。Piper 同样是留出目标，但不用于替代预注册主门。

CfC/液态网络本身不是本项目创新。研究对象是把低秩本体时间调制限制在硬共享液态 critic 的时间常数中，并结合事件可达率后验，在不做目标 TD 的条件下尝试传输事件价值。

## Stage 0 与 Stage 1

历史阶段的完整参数和口径以对应运行文档为准。已有数据环境中可运行：

```bash
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python scripts/run_stage0_experiment.py
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python scripts/run_stage1.py
```

Stage 1 是历史基线，保留了当时的固定事件模板和 MSE 头。Stage 2 已修复这些问题；不要用 Stage 1 脚本重写 Stage 2 结论。
