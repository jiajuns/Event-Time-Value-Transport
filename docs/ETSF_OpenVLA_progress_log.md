# ETSF × OpenVLA-OFT 进度日志

本文件是 OpenVLA-OFT 接入 ETSF 的唯一持续进度文档。每次实质代码改动、正式运行、问题定位、协议变更和门控结论都在这里追加；原始 rollout、模型和大型结果只保存在运行服务器，不提交 Git。

## 当前快照

- 更新时间：2026-08-26 17:34（Asia/Shanghai）
- Git 分支：`main`
- 最新实现提交：`ab9bb70a3aee968816449e7c4dac4fa543580194`
- 正式结果记录提交：`5a67786f1857bf3390e44ef37d510b40da691ea9`
- 运行服务器：`user@100.115.128.14`
- GPU：NVIDIA GeForce RTX 4090 D，24 GB
- 任务：RoboTwin `move_can_pot`
- 本体：Piper 双臂，间距 0.6
- OpenVLA-OFT checkpoint：`/home/user/etsf_openvla_models/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot`
- 正式 rollout 目录：`/home/user/etsf_openvla_rollouts_move_can_pot_20260826`
- 正式采集：150/150，成功 18、失败 132、带 canonical chain gap 10，审计通过
- OpenVLA on-policy 成功率：`18/150=12.0%`
- 正式训练输出：`/home/user/etsf_openvla_shadow_trained_move_can_pot_20260826`
- 本地数据备份：`/home/jj/etsf_openvla_rollouts_move_can_pot_20260826`（57 MB，150 个 HDF5）
- 本地训练备份：`/home/jj/etsf_openvla_shadow_trained_move_can_pot_20260826`（12 MB）
- 离线门结果：`action_ranking_authorized=false`
- 当前控制权限：ETSF 仅 shadow；没有修改 OpenVLA 动作，也没有候选动作排序权限

## 已完成

### 2026-08-26：OpenVLA-OFT B0 复现

- 在 RTX 4090 D 上加载 RLinf OpenVLA-OFT `move_can_pot` SFT checkpoint。
- 模型参数量 7.558B，BF16，14 维动作，25-step action chunk。
- 20 个官方 eval seed 成功 4 条，B0 成功率 `4/20=20%`。
- 修复 Ray 环境错误注入的 Vulkan ICD 路径；当前 RGB-only eager/SDPA 路径可运行。
- 结果只证明 Piper 单任务 baseline，不能外推到其他任务或机械臂。

### 2026-08-26：ETSF shadow 接线

- 从 OpenVLA `_discrete_prediction` 的动作 token 前一位置提取 4096 维 hidden。
- 接到 ETSF `4096→96` bridge、语义头和 ClockLNN。
- hook 前后确定性动作逐元素一致；真实环境完成 8 个 chunk、200 步。
- 此阶段 shadow 为随机初始化，只证明旁路接口，不代表价值性能。
- 提交并推送：`d8cc2c578309fa200cf7215559f23a608c31bcd0`。

### 2026-08-26：正式 OpenVLA rollout collector

新增 `scripts/collect_openvla_etsf_rollouts.py`：

- OpenVLA 每 25 步查询一次，每个动作逐步送入环境，保证动作、位姿和事件时间对齐。
- 每条 HDF5 保存：
  - query hidden `[N,4096]` 与 query step；
  - OpenVLA action chunks 和逐步实际动作；
  - 每步物体位姿与 proprio；
  - 首帧、真实终止 RGB；
  - 真实终止观测额外前向得到的 `terminal_hidden[4096]`，其动作绝不执行；
  - 冻结事件、环境成败、instruction 和 seed。
- 失败末帧始终保留；文件临时写入后原子替换，manifest 支持断点续跑。
- collector 固定 Python、NumPy、Torch seed 为 `20260826`，场景使用 RLinf 官方 seed。
- 2 条失败冒烟的 hidden、动作、位姿、RGB 和事件完整性检查通过。
- 成功 seed `100100005` 冒烟通过，100 步成功并保存真实 `eK`。

### 2026-08-26：事件口径修复

发现 Stage 2 clean 专家数据的几何终点不能直接等同于 OpenVLA `demo_randomized` 环境的成功触发时刻：

- 专家成功末端 `can` 距桌面约 1 mm；
- 部分 OpenVLA 轨迹在环境成功触发时仍距桌面约 38–52 mm；
- 因而一些真实成功轨迹没有触发冻结几何 `e3/e4`。

处理原则：

- `e1–e4` 继续使用 Aloha/ARX clean 数据冻结的几何阈值，不用 OpenVLA 标签重标定；
- 不伪造缺失的 `e3/e4`；
- 环境真实成功事件 `eK` 独立保留，即使中间几何事件缺失；
- 文件记录 `canonical_chain_has_gap`，训练和报告区分完整链与带 gap 的真实结局。

### 2026-08-26：shadow trainer 与离线门

新增 `scripts/train_openvla_etsf_shadow.py`：

- OpenVLA 全冻结，只训练 481,965 个参数：4096→96 bridge、共享 GRU 语义编码器、事件/可达/成功头和隔离 ClockLNN。
- ClockLNN 接收 `stop-gradient(semantic)`，时钟损失不能改写语义空间。
- 失败轨迹进入可达、成功分类和同事件 pairwise ranking；失败轨迹不进入 ClockLNN 时长损失。
- 按 episode 分层冻结 `100 train / 25 validation / 25 test`，不允许帧级泄漏。
- 训练 5 个初始化，仅用 validation 选择 checkpoint，test 只在选择后评估。
- 同事件 AUC 只累计同一事件内正负配对，不混入跨事件进度差；事件计数器在此条件下基线为 0.5。
- 数据审计会验证 hidden、terminal hidden、动作、位姿、末帧、唯一 seed 和 `eK↔success`；任一失败则训练前止损。
- 动态自检已通过前向、反向、缺事件链、配对计数和 episode bootstrap。
- 实现提交并推送：`ab9bb70a3aee968816449e7c4dac4fa543580194`。

### 2026-08-26：150 条正式采集与训练结果

正式数据：

| 项目 | 结果 |
|---|---:|
| rollout | 150 |
| 成功 / 失败 | 18 / 132 |
| OpenVLA 成功率 | 12.0% |
| canonical chain gap | 10 |
| HDF5 审计 | 10/10 项通过 |
| train / validation / test | 100 / 25 / 25 |
| 三份成功数 | 12 / 3 / 3 |

最初 20-seed B0 的成功率是 20%，扩大到 150 个官方 seed 后为 12%，说明 20 条估计偏乐观。固定 test 只有 3 个成功，低于预注册的最少 4 个成功门槛；不为过门重划数据。

训练 5 个初始化、每个 3000 步。validation 同事件 micro-AUC：

| seed | validation 同事件 AUC | validation Clock MAE |
|---|---:|---:|
| 20260826 | 0.7130 | 20.34 |
| 20260827 | 0.5278 | 23.50 |
| 20260828 | **0.8148** | 20.14 |
| 20260829 | 0.6481 | 24.86 |
| 20260830 | 0.5926 | 16.43 |

按 validation 选择 seed `20260828`，只在选择后评估 test：

| test 指标 | ETSF | 基线 / 门槛 | 结果 |
|---|---:|---:|---|
| 同事件 micro-AUC | **0.8258** | 事件计数器 0.5 | 通过 |
| 同事件配对 | 132 | ≥50 | 通过 |
| episode bootstrap 95% 下界 | **0.7083** | >0.5 | 通过 |
| 同事件 Brier | 0.1310 | 事件率 0.1056 | 未通过 |
| Clock duration MAE | 16.65 | 事件中位数 12.50 | 未通过 |
| test 成功 / 失败 | 3 / 22 | 各至少 4 | 未通过 |
| 事件分类 accuracy | 0.9149 | 事件索引 oracle 1.0 | 诊断，不作门 |

门控结论：

```text
action_ranking_authorized = false
policy_effect_during_collection_or_shadow = false
next_action = keep ETSF in shadow mode and do not rank OpenVLA actions
```

解释：4096→96 bridge 和共享语义头已经在严格同事件条件下学到明显分支信号，这一部分不是事件计数器；但成功概率过度自信、ClockLNN 在仅 12 条训练成功轨迹上过拟合，而且确认集正例不足。当前 checkpoint 可作为开发 shadow，不可接管候选动作选择。

下一版仅能作为开发迭代，不能把已经打开的同一 test 再称为全新确认集：

1. 只用 validation 做 temperature calibration，修复排序好但 Brier 差的问题；
2. ClockLNN 改用与 MAE 对齐的 robust log-duration 损失，并保存 validation early-stop checkpoint；
3. 增加新的 on-policy 成功轨迹或预注册新确认 seeds，保证确认集至少 4 个正例；
4. 通过新的密封确认门之前继续禁止动作候选排序。

本地校验：

- `shadow_gate_summary.json` SHA-256：`addee2509acdf55d7d9afe2ff7ecfc41a9d99f9e08d489830f3436b5f720291e`
- selected checkpoint SHA-256：`e8d5c8e9c967b6876f6570e702eca17d9e7137675840d6e2cfa178cc4e871537`

## 当前正式运行

采集命令：

```bash
cd /home/user/etsf_stage0/RLinf
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
.venv_openvla_robotwin/bin/python \
  /home/user/etsf_stage2/collect_openvla_etsf_rollouts.py \
  --model-path /home/user/etsf_openvla_models/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot \
  --rlinf-root /home/user/etsf_stage0/RLinf \
  --robotwin-root /home/user/etsf_stage0/RoboTwin \
  --robotwin-code /home/user/etsf_stage0/RoboTwin_RLinf_support \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --output /home/user/etsf_openvla_rollouts_move_can_pot_20260826 \
  --limit 150 --overwrite
```

采集完成后训练命令：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_stage2/train_openvla_etsf_shadow.py \
  --data /home/user/etsf_openvla_rollouts_move_can_pot_20260826 \
  --output /home/user/etsf_openvla_shadow_trained_move_can_pot_20260826 \
  --steps 3000 \
  --seeds 20260826 20260827 20260828 20260829 20260830
```

## 离线 shadow 门

必须全部满足才会输出 `action_ranking_authorized=true`：

1. 至少 100 条完整 rollout；
2. 留出测试集至少 4 条成功和 4 条失败；
3. 至少 50 个同事件正负配对；
4. 同事件 AUC 高于事件计数器的 0.5；
5. episode bootstrap 95% 下界高于 0.5；
6. 语义 Brier 优于训练集逐事件成功率基线；
7. ClockLNN 时长 MAE 优于训练集逐事件中位数基线。

门控前以及任一条件未通过时：

```text
ETSF = shadow only
policy action modified = false
OpenVLA candidate ranking = disabled
```

## 当前边界与下一步

- 当前 checkpoint 只覆盖 Piper `move_can_pot`，本轮只能验证 OpenVLA hidden 上的单本体 shadow critic；不能宣称 OpenVLA critic 已跨机械臂迁移。
- 跨本体正式实验还需要其他机械臂的原生 OpenVLA-OFT policy/action adapter 与 on-policy rollout，按本体留一训练共享 critic。
- 当前下一步：采满 150 条 → 完整性审计 → 5 初始化训练 → 冻结测试集离线门 → 根据门结果决定是否实现受保护的候选动作评分。
