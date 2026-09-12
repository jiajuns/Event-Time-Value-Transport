# Event-Time Value Transport (ETSF)

Official artifact repository for **Cross-Embodiment Value Transport via Event Semantics, Reachability, and Execution Time**.

ETSF transfers the *value side* of a manipulation policy across robot embodiments. Instead of forcing different robots to share joint coordinates or action semantics, it represents a task as a canonical event chain, learns a shared continuous-time event-successor critic in simulation, and adapts that critic to a target robot using only compact reachability and execution-clock statistics. The resulting value signal is injected into a VLA through Event-AWR during post-training; deployment remains a plain SmolVLA policy.

## 核心思想

不同机械臂可以用不同的关节、速度和控制器完成同一个语义事件序列。ETSF 因此把跨本体差异拆成两个通道：

1. **事件可达性**：目标本体是否能到达事件 \(e_j\)，由每事件 Beta 后验 \(\rho_b(e_j)\) 表示；
2. **执行时钟**：同一事件链在目标本体上快慢不同，由一个正的时间尺度 \(\beta_b\) 表示。

共享事件表示和液态 CfC critic 在源域训练后冻结。目标本体只拟合 \(K\) 个事件可达性统计和一个 control-clock 标量，总适配维度为 \(O(K+1)\)，不使用目标域 TD 更新共享 critic。

```text
RoboTwin simulation trajectories
        │ canonical state27 + event boundaries + real Δt
        ▼
Frozen shared event-successor critic (GRU + CfC)
        │
        ├── per-event reachability posterior ρ_b(e_j)
        └── one embodiment clock scalar β_b
        ▼
Semi-Markov value transport and TD(λ) advantages
        │
        ▼
Event-AWR weighted SmolVLA flow-matching post-training
        │
        ▼
Deployment: plain SmolVLA only
```

## 方法

### 1. 规范事件表示

任务被写成事件链 \(E_g=(e_1,\ldots,e_K)\)。论文实验使用 27 维规范边界状态，包括目标相对位置、左右末端相对物体位置、物体位移、夹爪状态、物体四元数、五阶段事件 one-hot 和解析任务谓词。原始关节轨迹、本体 ID、机器人像素和原始速度幅值不进入共享仿真 critic。

典型放置任务的事件链为：

```text
Approach → Grasp → Transport → Release → Placed
```

### 2. 连续时间事件 critic

GRU 汇总离散事件历史，CfC 用真实物理时间推进液态隐藏状态。参考时间定义为

\[
\Delta t_k^{\mathrm{ref}}=0.5\Delta t_k,
\qquad
\widetilde{\Delta t}_k=\beta_b\Delta t_k^{\mathrm{ref}},
\qquad
\alpha_k=\exp(-\widetilde{\Delta t}_k/\tau).
\]

源本体固定 \(\beta_b=1\)，因此源时钟等于预先定义的 reference clock。\(\beta_b\) 同时调制有限差分状态速率和 CfC closed-form 更新，避免时间输入只成为一个旁路特征。

共享 critic 预测下一事件、事件后果、成功概率、下一边界状态、累计时间和下一事件持续时间。仿真训练联合 TD、Monte-Carlo、事件分类、成功校准、符号一致、排序、持续时间和状态运输等损失。

### 3. 跨本体价值传输

目标本体的每事件可达性使用 Beta 后验：

\[
\rho_b(e_j)\sim \mathrm{Beta}(a_{bj},b_{bj}),
\qquad
\mathbb{E}[\rho_b(e_j)]=\frac{a_{bj}}{a_{bj}+b_{bj}}.
\]

时钟参数使用一维后验 \(p(\beta_b\mid D_{\mathrm{adapt}})\)。显式 semi-Markov 递归为：

\[
\widehat V_b(e_j)=\widehat c_j+\rho_b(e_{j+1})
\mathbb E_{\beta_b}[\gamma^{\widehat D_j}\widehat V_b(e_{j+1})].
\]

该算子只组装目标本体的 value/advantage，不新增一套目标 critic 网络。

### 4. UMI 到真实机械臂

真实 Astribot S1 没有仿真的物体位姿真值，因此 UMI 观察接口使用腕部 RGB。UMI 阶段产生 111 维视频事件表示：96 维 CfC 历史加四个事件头的 15 维概率。S1 的 state25 只进入独立的 64 维 adapter，不进入共享 CfC 主干：

\[
V_\phi(s_t)=\tanh\!\left(W_2\,\mathrm{GELU}
\left(W_1[z_t^{\mathrm{video}},g_t^{\mathrm{S1}}]+b_1\right)+b_2\right).
\]

UMI 观察器在 85 条训练和 21 条验证轨迹上适配 2,000 步。随后冻结观察器，并用 episode 级五折交叉拟合产生 TD(\(\lambda=0.95\)) 优势，防止同一轨迹同时用于拟合和给自己打分。

### 5. Event-AWR 策略后训练

真实时间折扣为：

\[
\gamma_t=\exp[-\kappa(t_{n(t)}-t)],\qquad n(t)=\min(t+50,T-1).
\]

优势被压缩成保守权重：

\[
w_t=\operatorname{clip}\left(\exp\left(\frac{A_t}{3\,\mathrm{scale}(A)}\right),0.9,1.1\right),
\qquad
\mathcal L_{\mathrm{actor}}=\frac1B\sum_i w_i\mathcal L_i^{\mathrm{FM}}.
\]

critic、事件标签和 sidecar 只在训练时使用。部署时不加载这些模块，三相机、state25 和任务文本直接产生 \(50\times25\) 连续动作。

## 论文结果

### RoboTwin 跨本体价值评估

实验包含 600 条轨迹：Aloha 与 ARX-X5 各 270 条作为源本体，Piper 与 UR5-WSG 各 30 条作为目标本体；所有方法使用相同划分、五个 paired seeds 和每任务 5 条目标适配轨迹。

| Method | Bellman MSE ↓ | Ranking AUC ↑ | Sign consistency ↑ | Updated shared params | Adaptation |
|---|---:|---:|---:|---:|---:|
| Target-from-scratch | 0.073 | 0.611 | 0.458 | 12,097 | 8.17 s |
| Progress critic | 0.059 | 0.550 | 0.565 | 0 | 0 |
| Terminal prototype | 0.092 | 0.567 | 0.496 | 0 | 0 |
| Compact RL token | 0.061 | 0.622 | 0.568 | 0 | 0 |
| Paired plain critic | 0.074 | 0.627 | 0.403 | 12,097 | 7.29 s |
| **ETSF** | **0.037** | **0.740** | **0.816** | **0** | **0.15 s** |

连续时间路径在 source leave-one-body-out 上将平均 AUC 从 `0.6628` 提高到 `0.6945`。完整组件实验的 AUC 为 `0.627 → 0.644 → 0.740`。

### S1 三任务闭环验证

| Task | Plain SmolVLA | CfC-AWR post-trained | Absolute gain |
|---|---:|---:|---:|
| Place vegetable on plate | 12/20 | 18/20 | +30% |
| Place fruit on plate | 11/20 | 19/20 | +40% |
| Pick up pot lid and place beside pot | 15/20 | 19/20 | +20% |

## 仓库结构

```text
experiments/experiment2_ablation_fixed_20260901/
    V8 仿真训练、评估、HPC 入口和冻结协议
real_robot/cfc_awr_three_camera/
    水果/锅盖三相机 CfC-AWR 数据转换、预检、训练和验收入口
umi_cfc/
    UMI RGB-CfC、state25 独立 value adapter、蔬菜/水果/锅盖后训练代码
paper_results/v8/
    五 seed 最终指标、time/CfC 审计和适配记录
docs/
    方法、真实机器人流程与 Release 资产说明
```

一次性 probe、测试代码、缓存、原始视频、HDF5、训练日志和 checkpoint 不进入 Git 历史。正式模型通过 GitHub Release 提供。

## 快速复现 V8 仿真实验

仿真与结果导出所需的最小 Python 依赖列在 `requirements-simulation.txt`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-simulation.txt
```

真实机器人 actor 后训练依赖实验使用的 SmolVLA/LeRobot fork；其数据 ABI 和 HPC 环境由 `real_robot/cfc_awr_three_camera/` 中的 preflight 与 Slurm 文件绑定，不能仅靠上述仿真依赖运行。

V8 冻结协议位于：

```text
experiments/experiment2_ablation_fixed_20260901/FROZEN_PROTOCOL.json
```

HPC3 入口：

```bash
bash experiments/experiment2_ablation_fixed_20260901/hpc3/queue_fixed_v2.sh
```

核心脚本：

- `scripts/experiment2_fixed_core_v2.py`：模型、CfC 时间路径和损失；
- `scripts/train_experiment2_fixed_v2.py`：3000-step 训练；
- `scripts/evaluate_experiment2_fixed_v2.py`：目标本体评估；
- `scripts/audit_source_lobo_dt_gate_v8.py`：真实 \(\Delta t\) 与零时间输入审计；
- `scripts/audit_experiment2_integrity_v5.py`：五 seed、数据划分和 checkpoint 完整性审计。

数据不随 Git 仓库分发。运行前需要按 `FROZEN_PROTOCOL.json` 配置 600 条 RoboTwin 轨迹根目录。

## 三相机 CfC-AWR 后训练

水果和锅盖正式入口：

```bash
cd real_robot/cfc_awr_three_camera

# smoke
sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=5 train_three_camera_cfc_awr.slurm
sbatch --export=ALL,CFC3_TASK=pot,CFC3_STEPS=5 train_three_camera_cfc_awr.slurm

# 通过 smoke 后正式追加 10,000 步
sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=10000 train_three_camera_cfc_awr.slurm
sbatch --export=ALL,CFC3_TASK=pot,CFC3_STEPS=10000 train_three_camera_cfc_awr.slurm
```

蔬菜任务的 UMI-CfC、五折 value adapter 和 AWR 入口位于：

```text
umi_cfc/_internal/cfc_value_umi_s1_20260906/
```

服务器路径是原实验的审计绑定。迁移到新机器或替换源 checkpoint 时，应同时更新数据路径、模型 SHA256 和 preflight 合约，而不是只替换 `SOURCE`。

## Checkpoints

从仓库页面的 [Releases](https://github.com/jiajuns/Event-Time-Value-Transport/releases) 下载：

- `etsf-v8-simulation-checkpoints.tar.zst`：V8 五个正式 simulation seeds；
- `etsf-v8-umi-three-task-critics.tar.zst`：共享 UMI RGB-CfC 与三个任务的 value adapters；
- `etsf-v8-cfcawr-vegetable-three-camera.tar.zst`；
- `etsf-v8-cfcawr-fruit-three-camera.tar.zst`；
- `etsf-v8-cfcawr-pot-lid-three-camera.tar.zst`；
- `etsf-v8-training-curves-and-results.tar.zst`：五 seed loss、参数更新、图和最终表；
- `SHA256SUMS`：所有 Release 资产哈希。

每个 SmolVLA 资产都包含完整 `pretrained_model` 目录。部署时不要只复制 `model.safetensors`，归一化与反归一化 processors 也是模型契约的一部分。

任务文本必须与训练数据一致：

```text
put the vegetable to the plate
put the fruit to the plate
pick up the pot lid and place it beside the pot
```

## 范围与限制

- 当前跨本体核心实验迁移的是 critic/value，不是关节级动作；
- 真实机器人结果覆盖一个 S1 平台和三个任务；
- 事件谓词仍需按新任务校准；仅修改自然语言 prompt 不能自动定义新的目标关系；
- Event-AWR 是离线加权回归，不是在线探索，也不对多个反事实 action chunks 做 Q 排序；
- “无推理开销”仅表示部署不加载 ETSF，训练阶段仍需事件观察、价值拟合和 AWR 计算。

## Citation

```bibtex
@inproceedings{etsf2027,
  title={Cross-Embodiment Value Transport via Event Semantics, Reachability, and Execution Time},
  author={Anonymous},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2027}
}
```
