# ETSF Stage 3：语义/时钟解耦的事件价值传输

## 1. 目的与状态

Stage 3 验证一个更窄、可证伪的命题：跨机器人共享任务语义，只在独立时钟分支中估计目标本体的时间变化，再用显式半马尔可夫算子构造折扣事件价值。

当前状态为：

- 共享语义编码器、固定 support successor 头、隔离 ClockLNN 和 β 后验已经实现。
- 已在 RTX 4090 D 上完成 3000 步、5 个配对种子的训练。
- Piper、UR5-WSG 的机制门通过；原首端 AUC 门被证明定义不当，决策能力改为“未充分验证”。
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
