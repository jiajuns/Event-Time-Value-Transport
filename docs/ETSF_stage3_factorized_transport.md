# ETSF Stage 3：语义/时钟解耦的事件价值传输

## 1. 目的与状态

Stage 3 验证一个更窄、可证伪的命题：跨机器人共享任务语义，只在独立时钟分支中估计目标本体的时间变化，再用显式半马尔可夫算子构造折扣事件价值。

当前状态为：

- 共享语义编码器、固定 support successor 头、隔离 ClockLNN 和 β 后验已经实现。
- 已在 RTX 4090 D 上完成 3000 步、5 个配对种子的训练。
- 三本体留一、失败末帧和同事件排序已实现；留出 Piper/UR5 的同事件 AUC 为 `0.612/0.792`，但配对数不足，仍属探索性正信号。
- 不继续接入动作条件 critic；当前仍是 event-value transport，而不是完整 critic transfer。

## 2. 架构

```text
对象相对状态、任务 token、事件 token
                  │
                  ▼
       共享语义编码器 Eθ（GRU）
             ┌────┴────┐
             ▼         ▼
   固定 support         stop-gradient
   successor 头             │
                             ▼
                    ClockLNN(z, β_b)
                             │
                    duration distribution
             └──────────┬──────────┘
                        ▼
          EventTransport(Ψ, E[γ^D], ρ)
                        │
                        ▼
                 本体相关事件价值
```

约束：

- `β_b`、机器人 ID、关节状态和原始速度幅值不进入语义编码器。
- 时长损失不能回传并重写语义表示。
- `β_b` 只进入 ClockLNN。
- HL-Gauss 语义 support 固定为 `[0,1]`，不会随 β 改变。
- `E[γ^D]` 使用 7 点 Gauss–Hermite 积分，不使用错误的 `γ^E[D]` 近似。
- 时长 MAE 使用 LogNormal 中位数；后验适配按 rollout 内平均似然，避免长事件链占据更多权重。

## 3. 数据与 mask

源域：

- Aloha、ARX-X5。
- 每任务每本体前 40 条训练、后 10 条验证。
- 共 480 条训练、120 条验证，全部成功。

目标域：

- Piper、UR5-WSG。
- 每任务索引 `0–4` 的 5 条 rollout 用于适配。
- 每任务索引 `15–19` 的 5 条 rollout 用于测试。
- 因此 `N=5` 实际是每目标本体 30 条适配 rollout。

统一时钟 mask 为：

```text
finite_duration AND duration >= 5 AND canonical_next_event_reached
```

正式数据中：

| 划分 | 有限区间 | 干净区间 | 排除短区间 |
|---|---:|---:|---:|
| 源训练 | 1746 | 1244 | 502（28.75%） |
| 源验证 | 438 | 310 | 128（29.22%） |

只有真实失败且未到达下一事件的轨迹才进入右删失似然。成功但事件检测没有落到 `eK` 的轨迹不会被伪装成失败删失样本。

## 4. 训练目标

```text
L = L_semantic_HL
  + 0.1 L_success
  + 0.25 L_within_task_rank
  + L_duration_NLL
  + 0.25 L_censor_NLL
```

源训练中没有失败轨迹，所以 `L_within_task_rank` 的有效 pair 数为 0。代码保留该模块并在输出中明确记录 `ranking_supervision_available=false`，但不会把它描述为已训练的排序能力。

ClockLNN 使用 `[-0.7,0.7]` 的合成时间 warp 识别 β 坐标。Aloha 以 `β=0` 为参考，ARX 的 β 在源训练中学习。目标端冻结全部共享参数，只计算一维网格后验：

```text
p(β_b | D_adapt) ∝ p(D_adapt | β_b) p(β_b)
```

若后验过宽、质量集中在支持边界或有效观测不足，推理自动回退到 `β=0`。

## 5. 指标

主要机制指标：

- 干净事件区间上的 duration MAE。
- 真实留出轨迹上的 event Monte Carlo return MSE。
- 目标 β 后验均值、标准差和回退状态。

主要决策指标：

- 同任务、同一非终止事件上的 success/failure 条件 AUC。
- 有效正负配对数和定义良好的任务×事件组数。
- 同一初始状态有重复 rollout 时的成功概率 Brier/NLL 校准。
- 成功轨迹内部的进度配对准确率必须与只使用事件索引的基线比较。
- 首端单 rollout AUC只作诊断，末端 AUC因 `eK` 泄漏而禁止用于门控。

Bellman self-residual 不再是 Stage 3 的主要正确性指标。当前事件头没有动作输入，因此也不再把相邻事件 TD 增量称为策略 advantage。

## 6. 正式运行

模块自检：

```bash
PYTHONPATH=/home/user/etsf_stage2 \
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  /home/user/etsf_stage3_code/run_stage3.py --stage self-test
```

正式训练：

```bash
PYTHONPATH=/home/user/etsf_stage2 \
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  /home/user/etsf_stage3_code/run_stage3.py \
  --stage main \
  --steps 3000 \
  --seeds 5 \
  --adaptation-count 5 \
  --data-root /home/user/etsf_stage1 \
  --output-root /home/user/etsf_stage3_run_20260826_v3
```

重新生成开发门：

```bash
PYTHONPATH=/home/user/etsf_stage2 \
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  /home/user/etsf_stage3_code/run_stage3.py \
  --stage gate \
  --output-root /home/user/etsf_stage3_run_20260826_v3
```

## 7. 正式开发结果

后验均值在种子间稳定地区分两种目标机械臂：

- Piper：约 `0.172–0.190`。
- UR5-WSG：约 `0.377–0.468`。

后验预测的 5 种子均值：

| 目标本体 | duration MAE：β=0 → 后验 | event MC MSE：β=0 → 后验 | goal MC MSE：β=0 → 后验 | 首端 AUC（诊断） | 同事件分支 AUC / 对数 |
|---|---:|---:|---:|---:|---:|
| Piper | 20.237 → **8.659** | 0.026442 → **0.023979** | 0.079970 → **0.077707** | 0.5417 | 0.4333 / 30 |
| UR5-WSG | 20.334 → **8.593** | 0.027491 → **0.022622** | 0.080178 → **0.073217** | 0.5042 | 0.4361 / 26 |

两个本体的 duration MAE 和 event MC MSE 均在 5/5 个种子上改善，因此：

```text
mechanism_gate_passed = true
decision_gate_passed = null
decision_gate_status = inconclusive_missing_source_failure_supervision
stop_before_critic_integration = true
```

原决策门取每条 episode 的第一个事件边界，实际询问初始状态能否预测一次随机 rollout 的最终成败，不是状态价值期望的可靠验收。末边界虽然得到 Piper `1.000`、UR5 约 `0.967` 的 AUC，但成功轨迹几乎总以 `eK` 结束、失败轨迹从不以 `eK` 结束，因此该数值包含事件标签泄漏并被禁止用于门控。

修正版同时报告：

- 首端 AUC：仅作诊断，5 种子均值接近机会水平。
- 同任务＋同非终止事件 AUC：控制事件进度后比较未来成败，但当前只有 Piper 30 对、UR5 26 对。
- 成功轨迹内部进度排序：Piper `0.9922`、UR5 `0.9795`；事件索引基线为 `1.0`，所以它只证明事件价值与规范链进度一致，不能证明细粒度失败判断。
- 倒数第二和末边界 AUC：仅作泄漏审计，不进入决策门。

## 8. 结论边界与下一步

当前能够支持：

> 一个隔离的低维时钟后验可以区分 Piper 与 UR5-WSG，并稳定改善目标机械臂上的事件时长预测和折扣事件 MC 价值误差。

当前不能支持：

> 不同机械臂已经能够直接共享完整 critic 或策略。

当前也不能表述为“价值排序已经失败”。更准确的状态是：失败分支判断尚未获得足够且无泄漏的训练和测试证据。

主要缺口是源训练没有失败轨迹，语义头没有成功/失败排序监督；现有 Piper/UR5 也已用于方案开发，不能再次充当确认性盲测。继续工作至少需要：

1. 重新执行 Stage 0 历史失败 seed，并保存完整物体状态与动作；历史 `raw.csv` 只有状态和步数，不能直接加入训练。
2. 为 Aloha、ARX 收集未筛选失败 rollout，并启用同任务、同事件、匹配初始条件的排序损失。
3. 增加第三个源本体或真实 leave-one-body-out 训练。
4. 使用全新本体或密封的新 rollout 做确认测试。
5. 只有决策门通过后，才将 `V_ETSF(s)` 接入 RoboTwin 的动作条件 `A(s,a)` 分支。

## 9. 三本体留一与失败终止状态

### 9.1 为什么事件索引准确率不能作为塌缩门

成功轨迹内部的事件索引天然严格递增，因此事件计数器的进度配对准确率已经是 `1.0`，任何模型都不可能“超过”该基线。`0.992<1.0` 只能说明模型在一个已被事件索引完全解决的任务上略有误差，不能单独证明或否定状态价值。

真正可证伪的塌缩指标是：固定任务和事件后，事件查找表对所有状态输出常数，其 success/failure AUC 必为 `0.5`；完整状态模型若包含事件之外的信息，应在完全留出的机械臂上超过该基线。

### 9.2 零成本三本体留一协议

```text
折 1：Clock={Aloha, ARX}，语义失败来源=UR5，完全留出=Piper
折 2：Clock={Aloha, ARX}，语义失败来源=Piper，完全留出=UR5
```

- 被留出的本体不进入共享训练，只使用每任务前 5 条适配 β/ρ，最后 5 条测试。
- 辅助本体的前 15 条训练、后 5 条验证，只为共享语义头提供混合成败监督。
- ClockLNN 两折都只使用 Aloha、ARX，避免把目标失败时长混入本体时钟识别。
- 训练 batch 对成功/失败 episode 均衡采样。
- 排序损失匹配任务和事件，比较每条轨迹在该事件下的最后观测。

### 9.3 失败末帧修复

旧事件抽取器只保留“已经到达的规范事件帧”。若轨迹未到 `e12` 就失败，记录中只剩初始 `e0`，真正显示抓空、停滞或回退的末帧被丢弃。仅把失败 episode 加入 DataLoader 并不能提供失败状态监督。

修复后，失败轨迹追加一个终止观测边界：

- 输入为最后一帧对象状态；
- event token 沿用最后已到达的规范事件，不输入失败标签；
- 同一事件存在多次观测时，排序和评估使用最后观测；
- 对从未推进出 `e0` 的失败，成功轨迹初始 `e0` 与失败轨迹终止 `e0` 构成状态质量配对。

这使可训练的匹配事件对从：

| 留出本体 | 不含失败末帧 | 含失败末帧 |
|---|---:|---:|
| Piper | 872 | 4766 |
| UR5-WSG | 3260 | 7870 |

### 9.4 正式 LOEO 结果

RTX 4090 D，两折各 3000 步×5 种子：

| 留出本体 | 语义失败来源 | duration MAE：β=0→后验 | event MC MSE：β=0→后验 | success-only 同事件 AUC | LOEO＋失败末帧 AUC |
|---|---|---:|---:|---:|---:|
| Piper | UR5-WSG | 14.727→**4.162** | 0.026325→**0.024835** | 0.4333 | **0.6119** |
| UR5-WSG | Piper | 18.066→**8.656** | 0.032681→**0.029241** | 0.4361 | **0.7917** |

Piper 的 5 个种子同事件 macro-AUC 为 `[0.6905, 0.5000, 0.6190, 0.6190, 0.6310]`；UR5 为 `[0.8333, 0.7917, 0.9167, 0.6250, 0.7917]`。失败监督在 UR5 折上高度稳定，在 Piper 折上仍存在本体差异。

```text
mechanism_gate_passed = true
decision_gate_passed = null
decision_gate_status = inconclusive_insufficient_matched_event_pairs
event_only_same_event_auc_baseline = 0.5
```

当前最准确的结论是：事件查找表退化已被定位，混合失败监督和失败末帧能在完全留出的另一机械臂上恢复事件内状态排序；但留出测试仍只有 Piper 30 对、UR5 26 对，且两种本体均已用于开发，因此需要全新密封 rollout 或新本体确认。

### 9.5 关于 RoboTwin 随机化数据

RoboTwin 官方 `demo_randomized` 主要随机化场景杂物、光照、纹理、桌高和相机。官方采集流程会先搜索可成功的 seed 再回放保存 demonstration，并未承诺随机化集含失败 episode，也没有固定“每任务×本体 100 clean＋400 failure-rich randomized”的规格。当前服务器未下载随机化目录，因此不能将它当作现成失败源。
