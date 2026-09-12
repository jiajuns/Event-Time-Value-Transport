# RoboTwin 多任务 GNN+CfC 事件世界模型（实验分支）

本目录是 `gnn-cfc-event-world-model` 分支的动作条件事件模型实现，**不是** ICRA V8 正式五 seed critic，也不是已经完成的多任务预训练 checkpoint。

## 结构与接口

```text
typed scene graph ── 3-layer message-passing GNN ──┐
physical Δt ────────── continuous-time CfC ────────┤
goal relation graph + task language ────────────────┼─ future relations / events / goal / risk
candidate chunk: 14-D canonical dual-EE effects ───┘
```

共享模型读取公开的物体/容器/支撑/可动部件/夹爪节点、关系边、真实时间差、目标关系和规范化的双臂末端动作。**不读取**机器人 ID、原生关节编号或 S1 state25。`universal_event/model.py` 可以按形状与语义严格匹配，从 UMI Event V4 初始化 CfC 和三个公共关系头；迁移 12 个张量不等于整个世界模型已训练。

`universal_event/schema.py` 定义 12 种任务目标、12 类关系、10 类事件和 8 类节点。语言编码器是轻量 byte-level GRU；它没有被验证为开放世界语言理解。模型输出是预测/评分，不是物理接触真值或已校准 Q 值。

## 已完成的工作与边界

- 24 条小规模 clean 仿真矩阵覆盖 12 个任务、5 个本体；22 条通过轨迹质量门，2 条保留为专家真实失败。审计记录在 `artifacts/small_simulation_final_20260909/`，原始 HDF5 不随代码仓库分发。
- HPC `smoke_607801` 在已有 `move_can_pot` 数据上完成 10 optimizer steps：532 条 train、68 条 validation；从 Event V4 精确复制 12 个兼容张量，证明前向/反向/保存链路可运行。它**没有**在 24 条多任务矩阵上完成正式预训练，也没有跨任务或跨本体效果结论。
- 三 seed、100,000-step 的 `hpc/slurm_train_3seed.slurm` 是运行入口模板；当前目录没有对应的正式训练结果，不能将 smoke loss 当作正式模型表现。
- `infer.py` 的 `UniversalEventChunkScorer` 接口能比较候选 14-D 动作 chunk，但尚无已验证的真实 S1 策略接入。

## 代码

- `collect_robotwin.py`：RobotWin 专家轨迹到规范图/动作 HDF5；
- `run_small_simulation.py`：小规模任务×本体矩阵；
- `audit_dataset.py`：格式、事件覆盖、身份与质量审计；
- `train.py`：episode 级 train/validation/held-out 划分、目标关系与动作条件预测训练；
- `infer.py`：候选 chunk 评分；
- `universal_event/{schema,model}.py`：事件图 ABI、GNN、CfC、语言/目标/动作编码与后果头。

在具备 RobotWin 环境和规范 HDF5 数据的机器上，训练入口为：

```bash
python train.py \
  --data /path/to/universal_event_hdf5 \
  --output /new/output/path \
  --v4-checkpoint /path/to/umi_v4/selected_observer.pt \
  --steps 100000 \
  --heldout-body piper \
  --heldout-tasks open_microwave place_empty_cup
```

`--output` 必须是不存在的新目录；命令是运行示例，**不是**已完成实验的声称。正式跨本体评估还需要独立的留一本体/留一任务测试、动作敏感性和闭环成功率验证。

权重与回执在实验性 GitHub Release 中提供，详情见 [GNN 分支说明](../../docs/GNN_CFC_BRANCH.md)。
