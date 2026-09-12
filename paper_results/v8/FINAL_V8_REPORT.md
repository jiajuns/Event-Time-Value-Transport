# ETSF A/B/C 修复与 v8 最终实验报告（2026-09-01）

## 1. 最终结论

最终保留 v8。它在冻结的 600 条轨迹、同一目标留出集、5 个 seed、每个正式模型严格
3000 optimizer step 的条件下，使 A+B+C 在三项模型质量指标上均优于 A+B，并且同一批
A+B+C `step_3000.pt` 同时用于消融表和横向表。

| 模型 | Bellman MSE↓ | 排序 AUC↑ | 符号一致率↑ | 成功率 MAE↓ | 更新参数 | 适配 s |
|---|---:|---:|---:|---:|---:|---:|
| A | — | 0.626834 | 0.646809 | 0.483215 | 0 | 0 |
| A+B | 0.091820 | 0.643855 | 0.597163 | 0.490748 | 0 | 0.000278 |
| A+B+C (v8) | **0.036830** | **0.740012** | **0.816312** | 0.490748 | 0 | 0.151631 |

相对 A+B，v8 的 MSE 降低 59.9%，AUC 提升 0.09616，符号一致率提升 0.21915。
相对旧 TABLE II，v8 同时超过旧最佳 MSE 0.059374、AUC 0.6383 和符号 0.6693。

不能声称 A+B+C 的正式成功率 MAE 严格最优。冻结协议规定所有事件模型都使用同一个
N=5 目标适配集的 `Πρ_j`，因此 A+B 与 A+B+C 必然同为 0.490748，与模型权重无关。
A 的 0.483215 来自低概率退化读出，其 Brier 为 0.368242；v8 的校准 success-head Brier
为 0.273533，但这不是论文冻结的正式 MAE 读出。不得为了制造“全指标最优”切换口径。

## 2. 最终模型与训练设计

- A：共享 canonical `state27`，真正忽略 padding 的 GRUCell 编码器，轨迹成功头。
- A+B：A 加 5 个事件价值头、后继事件头、真实下一边界 TD、MC 锚点、事件方向损失、
  `(body,event)` 桶内 listwise 排序。
- A+B+C：A+B 加不可旁路的 CfC 时间动力学。CfC 输入为事件边界状态、事件 one-hot 与
  有限差分状态导数 `(x_t-x_{t-1})/dt_eff`；同一个 `dt_eff=0.5*beta*真实dt` 驱动状态
  导数和 CfC closed-form 更新。时间路径单独预测下一边界状态、累计时间、下一段持续时间
  和事件价值，完整价值再加入有序事件进度约束。

CfC 没有可学习的零门控，也没有 A+B 回退。成功率头独立读取 GRU 路径，避免目标本体
时钟校准污染轨迹成功概率。

训练阶段固定为：step 1–2000 事件模型预训练，step 2001–3000 排序微调。正式 checkpoint
固定读取 step 3000；source-LOBO 只用于 success-head 校准与审计，不早停、不选目标测试。

## 3. 连续时间是否真正发挥作用

v5/v6/v7 均未通过时间反事实审计，虽然完整模型主表已有提升：它们在目标域将 Δt 置零后
AUC 更高。因此这些版本全部作为失败中间结果保留，没有定稿。

v8 增加显式状态导数后，5 个 seed 在两个 source-LOBO 留出本体上的真实 Δt 排序增益均为
正：`+0.02358/+0.02012/+0.02759/+0.02231/+0.06502`。5-seed 平均 AUC 为
0.694527，Δt=0 为 0.662802，增益 +0.031725。

冻结配置后进行目标评估：真实 Δt AUC 0.740012，Δt=0 AUC 0.737751，增益 +0.002262。
真实 Δt 对 MSE 有 0.000095 的小代价，对符号一致率有 0.041135 的代价；所以严格结论是
连续时间信息改善排序，而完整模型的 MSE/符号总增益主要还来自状态运输监督与事件进度
约束。不能写成“Δt 单独改善全部指标”。

动力学损失的 5-seed 正式训练首/末 100 step 均值：

| 辅助目标 | 首 100 | 末 100 | 下降 seed 数 |
|---|---:|---:|---:|
| 下一持续时间 | 0.141063 | 0.020233 | 5/5 |
| 累计时间 | 0.163713 | 0.001874 | 5/5 |
| 下一边界状态 | 0.733159 | 0.112054 | 5/5 |
| 时间 MC | 0.147416 | 0.085906 | 5/5 |

`time_input/time_sequence/time_output/time_value_head/time_state_head/elapsed_head/duration_head`
各参数组在 5 个 seed 中均有非零更新；CfC 不是未训练的装饰模块。

## 4. 实验完整性

- 数据：Aloha 270 + ARX 270（源），Piper 30 + UR5 30（目标），总计 600。
- 数据 roster SHA-256：`a71666e8297c1b3919c2ec76de3e6b4e2a022adba0889d861f7435895b89d971`。
- source z-score SHA-256：`f84434f276e8a4f118e76b35c061e72af04a662a58df613503ebfb04041f5d14`。
- 5 个 seed：20260901–20260905。
- 每个正式 run：step 0–3000 共 3001 个 snapshot，另有正式 `step_3000.pt`。
- 15 个 A/B/C run 均验证为 3000 step、目标零训练/零选模泄漏、退出码 0。
- 5090 的 8 项契约测试、HPC3 全 600 数据 smoke、最终完整性审计全部通过。
- 代码清单摘要 SHA-256：`b77d307b776fbd0459f217c07d5435248d6016694bc235be39a67b51113a0262`。
- HPC3 队列只重训发生变化的 A+B+C v8，并在提交前硬检查已审计的 10 个 A/A+B
  checkpoint；每个 seed 的 source-only 真实 dt AUC 门失败时作业非零退出，目标域评估不会启动。

## 5. 最终文件位置

HPC3 实验根目录：

`/data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901`

最终表与审计：

- `results/final_v8/ablation_summary.csv`
- `results/final_v8/ablation_raw_metrics.csv`
- `results/final_v8/horizontal_with_same_abc_summary.csv`
- `results/final_v8/integrity_and_cfc_use_audit.json`
- `results/source_lobo_dt_gate_v8_seed<seed>.json`

最终 A+B+C checkpoint：

- `results/train_v8/e2_cfc_event_fixed/seed20260901/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260902/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260903/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260904/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260905/formal/step_3000.pt`

对应 SHA-256：

- seed01 `90f8e7263bed20c4c390ea9270d1dbbca63a10d4e8eaf11f58aa902615397124`
- seed02 `3f587b4efb26950251f15f50848ec1e32b22317b7856214a387cbb77693ef5eb`
- seed03 `d7796de76e7cf5dc76fa5a052295092112469b114cffbb54e00cbc671ff05735`
- seed04 `d9f4516dea800f247060700d43199b45a0c94878157c02d81da3f9788bcd9ee6`
- seed05 `f1390692c239a9cf6d467b51b8eede6567842a307d30ee427736a74b9190ea58`

注意：以上 checkpoint SHA 是在 v8 正式训练完成后计算；同一文件被消融表与横向表共同引用。
