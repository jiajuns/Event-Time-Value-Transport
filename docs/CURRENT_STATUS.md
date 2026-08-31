# 当前状态

> 最近核对：2026-09-01 01:49 CST
> 服务器：`user@100.115.128.14`，NVIDIA GeForce RTX 4090 D。

## 当前结论

4090 上已经重新启动正式主线。当前不是在训练参数，而是在用**官方 RoboTwin SmolVLA checkpoint**采集 `move_can_pot / Aloha / clean+randomized` 的四候选真实后果监督；400 个 decision groups 完成后会自动训练五成员 v14 Liquid-CfC 共享头，再自动在同一个官方 SmolVLA、同一个 `move_can_pot` 和同一组 100 个 clean seeds 上做 candidate-0 与共享头 best-of-4 的配对闭环对比。

旧的 EE16 自定义 actor 数据和 2026-08-31 失败流水线均保留但隔离，**没有混入本次官方 SmolVLA 训练**。

| 阶段 | 状态 |
| --- | --- |
| 官方 SmolVLA 14-D ABI | 已核实；真实 preprocessor state 和动作均为 14-D，GPU 前向输出 `[1,50,14]` |
| 关节候选到共享头 | 已修复；官方 `joint14` 经解析 FK 变为 canonical EE16，原生执行仍用 `qpos` |
| 单臂约束 | 使用任务场景的 `task.arm_tag`，另一只手臂保持当前关节与夹爪状态；基线和共享头共用 |
| 真实 smoke | 完成 1 个四候选 group；全部数组有限；candidate-0 失败、candidate-2 成功，存在 `+1` oracle headroom |
| 正式源采集 | 已启动，目标 `400 decisions / 1600 branches` |
| v14 五成员训练 | 等待完整 manifest，随后自动启动，每成员 3000 step |
| 同任务配对对比 | 等训练完成后自动启动，100 个固定 clean seeds |
| 四个留出本体 | 本轮不参与训练；完成同任务 Aloha 对比后再做 ARX/Franka/Piper/UR5 迁移评估 |

## 当前远程进程

```text
pipeline PID: 674735
首个 collector PID: 674739
pipeline PPID: 1
status: filling_aloha_source_stratum_quota
active: clean / query 0 / seeds 2026091000..2026091003
```

流水线与本机、当前 SSH 会话无关，本机可以关机。采集主要受 SAPIEN 物理仿真限制，`GPU-Util=0%` 的瞬时采样不等于停机；以 `pipeline_state.json`、子进程和 manifest 组数为准。

单个真实 smoke group 用时约 103 秒。按该样本粗估，400 组采集约 11.5 小时，加上模型重复加载、五成员训练和 100-pair 闭环，完整结果大致需要 12–15 小时；这只是运行时估计，不是完成承诺。

## 数据与训练合同

每个根状态由冻结 SmolVLA 通过确定性 antithetic flow noise 产生 4 个候选。四个候选都从同一个可恢复根状态独立执行到 200-step 终点，并保存事件、时长、成功/失败、恢复、对象 SE(3) 效果、终局阶段进度和不确定性所需监督。数据包含成功和失败后果，不是只学习成功轨迹。

```text
官方 SmolVLA 观察/输出：left 6 joints + gripper + right 6 joints + gripper = 14-D
              ↓ 同一无学习单臂适配器
执行：RoboTwin qpos 14-D
评分：Aloha URDF FK → world EE16 → canonical action effect 14-D
              ↓
v14 Liquid-CfC 共享事件头（actor 权重始终冻结）
```

采集器现按每次最多 4 个 seed 启动一个独立进程。这样 SAPIEN/MPLib 对象会随子进程退出而完整释放，修复旧流水线运行几十个 scene 后的 pybind11 GIL/SIGABRT 累积崩溃；每个 group 原子落盘，异常后只续跑缺口。

## 自动执行顺序

```text
Aloha clean+randomized：400 decisions / 1600 real branches
  → 冻结官方 actor 绑定的 source manifest
  → 标签盲 requested-seed train/validation split
  → 5-member v14 Liquid-CfC（只读 Aloha source）
  → 冻结一个 ensemble checkpoint
  → Aloha move_can_pot clean100 配对闭环：
       official SmolVLA candidate-0
       vs 同一 actor 候选池 + v14 best-of-4
  → 输出 SR、DeltaSR、阶段进度、95% CI、McNemar 与选中候选分布
```

5090 上此前官方 SmolVLA 的原生 400-step `move_can_pot` clean100 结果为 `8/100`。它可作为外部名义基线，但本轮共享头使用冻结的 200-step execute50 协议，因此不会把 `8%` 与新结果直接相减；正式改进数字只取本轮同 seeds、同初态、同候选协议的配对结果。

## 权威路径

```text
GitHub commit（已 push 到 main）：
cb90a6f37a38797e56b5b119d897b0c1c3a5f2e9

4090 只读代码：
/home/user/etsf_robotwin2_v14_official_code_cb90a6f_20260901

正式流水线根目录：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901

实时状态：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901/pipeline_state.json

采集 manifest：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901/aloha_source_branches/manifest.json

训练状态与最终 checkpoint 摘要：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901/liquid_v14_training/training_state.json
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901/liquid_v14_training/training_summary.json

同任务配对结果：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901/paired_clean100_official_smolvla_vs_liquid/paired_report.json

总日志：
/home/user/etsf_robotwin2_v14_official_smolvla_aloha_20260901.pipeline.log

已完成的隔离 smoke：
/home/user/etsf_robotwin2_v14_official_smolvla_smoke_20260901/manifest.json
```

## 结果边界

本轮最终能回答“在 Aloha 的同一官方 SmolVLA 和同一任务上，共享头是否只通过重排提高成功率”。即使该结果为正，也还不能声称已经证明跨本体提升；跨本体主张必须继续在未参与训练的 ARX-X5、Franka、Piper 和 UR5 上运行同协议配对闭环。

以后继续更新本文件，不再新增时间戳式进度文档；服务器 JSON 和最终 report 是最高事实来源。
