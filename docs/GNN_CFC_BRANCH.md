# GNN+CfC 实验分支说明

分支：[`gnn-cfc-event-world-model`](https://github.com/jiajuns/Event-Time-Value-Transport/tree/gnn-cfc-event-world-model)。它保存两条相关但不同的实现，不应与正式 ICRA V8 critic 和 CfC-AWR 三任务策略混为一谈。

| 模块 | 输入与结构 | 已完成的训练/验证 | 不能据此声称 |
|---|---|---|---|
| UMI 关系观察器 V3 | 腕部 RGB + YOLO 辅助框 → 角色图、两层共享消息传递 GNN → CfC → 关系/事件/目标谓词 | 106 段 origin，85 train/21 validation，4,000-step 预算，选中 step 2,200 | 独立测试准确率、跨任务/跨本体迁移、校准 RL value |
| UMI 证据观察器 V4 | V3 + 物体/目标成对视觉、当前证据与遮挡 prior | 4,000-step 预算，选中 step 2,600；内部验证与失败回放已记录 | 可作为可靠成功奖励；V4 未通过冻结晋级门槛 |
| RoboTwin 动作条件世界模型 | 规范场景图 → 三层 typed GNN + CfC；语言、目标图和 14-D 双臂末端 action chunk 条件化未来关系 | 24 条小规模矩阵已审计；另在单任务 `move_can_pot` 数据上完成 10-step smoke | 已完成大规模多任务预训练、零样本跨本体、真实机器人闭环成功 |

UMI V3 的内部 validation CE 为 `0.4711`，同预算 no-graph 对照为 `0.5023`。在 1,200 个已标目标失败点中，V3 误报 3 次、no-graph 误报 5 次；但 V3 成功召回 `68.21%`，低于 no-graph 的 `75.50%`。这只是 AI 标注的内部验证，不是独立测试指标。

V4 在同一内部参考的 1,200 个已标失败点中假成功为 0、151 个已标成功点召回 `75.50%`，但预测 unknown 比例 `73.40%`，超出预注册门槛。已知盘外录像回放仍有 11 个 false-confirmed-success 点，因此 V4 未晋级默认模型，更不能直接提供 RL 奖励。它的遮挡 prior 是短时关系推断，非生成未来画面，也未做动作条件预测。

## 代码定位

- UMI 图与 CfC：`umi_cfc/_internal/event_rl/relational_graph.py`、`relational_observer.py`、`evidence_observer.py`；
- UMI 数据准备/训练/推理：`umi_cfc/_internal/scripts/` 中的 `build_relational_*`、`train_relational_observer_v3.py`、`train_evidence_observer_v4.py`、`predict_*`；
- RoboTwin 多任务动作条件模型：`experiments/robotwin2_universal_event_pretrain_v1_20260909/`。

公开分支只包含代码、协议和小型审计回执，不包含用户原始 UMI/RobotWin 视频、HDF5、YOLO 第三方权重或训练缓存。训练 GNN 观察器需自行准备与现有 schema 对齐的缓存及 AI 标注；直接使用本分支中的训练脚本时，必须传入 `--data/--graph/--annotations`（V4 还需 `--visual/--evidence`），否则不能凭一个 checkpoint 重现训练。

```bash
# V3：在 umi_cfc/_internal/scripts 下运行
python train_relational_observer_v3.py \
  --data /path/to/origin_events.npz \
  --graph /path/to/origin_graph.npz \
  --annotations /path/to/relations_ai_reviewed.csv \
  --output /new/v3_run --device cuda

# V4：在同一目录下运行
python train_evidence_observer_v4.py \
  --data /path/to/origin_events.npz \
  --graph /path/to/origin_graph.npz \
  --visual /path/to/origin_pair_visual.npz \
  --annotations /path/to/relations_ai_reviewed.csv \
  --evidence /path/to/training_evidence_annotations.json \
  --output /new/v4_run --device cuda
```

实验性 Release：[`gnn-cfc-experimental-20260913`](https://github.com/jiajuns/Event-Time-Value-Transport/releases/tag/gnn-cfc-experimental-20260913)。其中只放 UMI V3/V4 已训练观察器与 RoboTwin 10-step smoke 权重；每份都有训练/选择回执和 SHA256。**RoboTwin smoke 权重只用于 API 与流程检查，不建议用于动作排序或部署。**

与主线衔接的正确步骤是：先在正式多任务数据上训练和留一本体/任务验证世界模型，再把动作条件预测转成有独立验证的价值/优势，最后才能接入 SmolVLA 后训练。当前分支没有完成这三个步骤。
