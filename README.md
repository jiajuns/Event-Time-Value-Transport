# Event-Time Value Transport

Event-Time Value Transport（ETSF）研究跨机器人本体的 critic/value 迁移：把动作候选的后果分解为规范事件、竞争风险时长、成功/失败/恢复、对象状态变化与不确定性，再用外置共享头重排冻结 VLA 的候选动作。

## 当前主线

当前正式实验是 RoboTwin2.0 `move_can_pot` 五本体 leave-one-body-out（LOBO）：

```text
冻结统一 actor 产生 N 个嵌套候选
             ↓
四个本体训练的共享事件 critic
             ↓
在第五个、未见 critic 标签的本体上重排
             ↓
比较 actor N1、ETSF N4/N8、RAC、WCM 和 candidate oracle
```

五个本体为 `aloha-agilex`、`arx-x5`、`franka`、`piper`、`ur5`；条件为 `clean`、`randomized`。每个 `body × condition` 使用 100 个预注册 paired seeds。

当前论文主张边界是：

> 证明外置 critic/event head 对未见 critic 监督的本体发生迁移，并提高已有冻结 actor 的任务成功率。

若 actor 训练时已经见过目标本体，不能把这个实验写成“actor 与 critic 联合零样本跨本体”。共享头的 AUROC、Brier、时长 MAE 等是机制指标；最终迁移结论必须来自完整五折的 paired `DeltaSR`。

最新远程接管信息见 [模型与远程队列交接](docs/PROGRESS_20260831_1047_MODEL_HANDOFF.md)。该文件记录的是接管时快照，不替代服务器最终 report；只有完整五折和闭环报告可以作为效果结论。

## 最新共享头

v13 输入的是跨本体 canonical state/action effect，不输入本体 ID。主要输出为：

```text
p(next event)
p(competing-risk holding time / censoring)
p(success / failure / recovery / regression)
p(object SE(3) effect)
aleatoric + ensemble epistemic uncertainty
```

单标量 utility 只负责部署时排序；结构化输出保留为监督、校准、诊断与跨本体机制证据。完整结构见 [v13 共享头设计](docs/ETSF_ROBOTWIN2_SHARED_HEAD_V13_CAUSAL_BALANCE_ZH.md)。

## 正式测评

最终测评口径见 [跨本体共享事件头正式测评协议 v3](docs/ROBOTWIN2_CROSS_EMBODIMENT_EVALUATION_PROTOCOL_V3_ZH.md)。最重要的两层是：

1. **闭环结论**：五折 paired `DeltaSR`、`DeltaStage`、requested-seed cluster 95% CI、cell 内 exact McNemar；
2. **机制解释**：候选覆盖/oracle regret、AP/Brier Skill、事件 macro-F1、删失感知时长指标、对象物理误差、risk-coverage 与推理成本。

当前同一 decision root 的 N≤8 动作分支称为 `CandidateCoverage@N`，不能称为外部工作里“独立完整 rollout”的 `pass@k`。

## 仓库导航

```text
configs/     早期阶段配置
docs/        设计、协议、交接与历史复现文档
scripts/     训练、采集、闭环、baseline、评估和 watcher
tests/       长期协议/数据契约回归测试（不是实验结果）
```

- [scripts 分类与当前生产依赖](scripts/README.md)
- [docs 分类与阅读顺序](docs/README.md)
- [五折数据闭包与基线协议 v1](docs/robotwin2_cross_embodiment_baseline_and_metrics_v1.md)
- [早期 Stage0–Stage3 运行手册](docs/ETSF_agent_runbook.md)
- [OpenVLA-OFT 技术路线](docs/ETSF_OpenVLA_OFT_full_technical_route.md)

`scripts/` 暂时保持扁平布局，因为 runner、`PYTHONPATH=scripts` 导入、远端命令和冻结文件哈希都依赖现有 basename。分类通过 `scripts/README.md` 完成，不为目录整齐而破坏当前队列。

## 当前生产链

当前主线按以下顺序运行：

```text
actor execute5/execute50 配对协议
  -> primary 五本体候选 branch materialization
  -> v13 五折共享头
  -> v13 N1/N4/N8 闭环与 oracle 报告
  -> RAC 五折与闭环
  -> WCM 五折与闭环
  -> 统一正式测评表
```

关键入口：

- `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- `scripts/train_robotwin2_five_body_lobo_relative_action_critic_v1.py`
- `scripts/train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py`
- `scripts/run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py`
- `scripts/evaluate_robotwin2_five_body_lobo_n1_n4_n8_oracle_v1.py`
- `scripts/materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py`

RAC 是相对动作标量 critic；WCM 是参数量匹配的 future-latent/world-model critic。两者是结构对照，不是 ETSF 的组成部分。

## 数据与结果不进 Git

仓库只保存代码、协议和小型配置，不提交：

```text
data/ features/ results/ logs/ rollouts/ checkpoints/
*.hdf5 *.npz *.pkl *.pt *.pth *.zip
```

官方 expert demonstration 用于 actor/状态参考；共享头的正负监督来自 simulator 中真实执行的同根候选 branch outcome。未标失败的 expert episode 不能自动当失败。

训练输出、checkpoint、日志和最终 receipt 位于 4090 服务器的实验输出根。不要把 watcher 的等待状态当作模型结果，也不要从本地缓存推断远程完成状态。

## 测试与临时文件

`tests/` 中已跟踪的文件是协议、哈希绑定、LOBO 防泄漏和 outcome schema 的长期回归契约，并非训练数据或实验结果。一次性 probe、调试脚本、pytest cache 和字节码不应保存在仓库工作区。

聚焦验证建议：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q -p no:cacheprovider <focused-test-files>
```

一次性输入和脚本放到 `mktemp -d /tmp/etsf-check.XXXXXX`，完成后删除；不要再创建 `scripts/*_probe.py` 或新的常驻临时 test 文件。

## 历史线

- Stage0–Stage3：事件/时间解耦与早期多本体机制实验；
- OpenVLA/OpenVLA-OFT：hidden-state bridge、OOF、structured heads 和 action reranking；
- SmolVLA/Piper schema5/schema6：单本体数据契约、paired-success 和 observer 线；
- RoboTwin2 v6/v8/v9/v10/v12：共享头设计演化。

这些代码和文档用于复现与方法演化，不是当前 v13 运行入口。精确分类见 [scripts 索引](scripts/README.md) 和 [docs 索引](docs/README.md)。
