# 共享事件头 v13 远程续跑记录（2026-08-31 07:55 CST）

## 当前结论

- 4090 上尚未开始五折共享头训练；当前先完成 actor candidate-0 的 execute5/execute50 正式配对，
  以冻结后续所有数据采集和闭环评估唯一允许的动作执行时域。
- 记录时完成 `42/200` 对、`84/400` rollout，状态 `running`，已经从 aloha-agilex 进入 arx-x5。
- 先前 `38/200` 对时 execute5 为 `25/38 = 65.8%`，execute50 为 `20/38 = 52.6%`；该数值是
  未完成、顺序不均衡的中间结果，不能用于冻结协议或声称共享头有效。
- 共享头已经升级为 v13，但真实跨本体效果仍必须等待五折 LOBO checkpoint 与五本体 N=1/4/8
  完整配对结果。

## 代码与模型版本

- GitHub 主分支提交：`57bd63f5a5f325f3c736075d38eeebaf69338cc3`；
- 模型族：`terminal_consequence_utility_shared_event_head_v13`；
- 只读远程代码：`/home/user/etsf_robotwin2_v13_protocol_code_57bd63f`；
- watcher SHA-256：`83aed676de2e0ad9179e7647255e32862b72fbd2488beb5945cc37551d519bf0`；
- trainer SHA-256：`c0070c540efa76e91bd6594684db0662c595b134c7d5a2962eba3d5488ea44b4`。

v13 的实质改动是 source-only `(body, condition, current_event)` 平方根逆频率 proper-loss
平衡，不改变参数张量、forward 或 candidate ranking 接口。正式 LOBO watcher 会显式传入
`causal_body_condition_event_sqrt`，并拒绝 v12、empirical mode 或任何 `heldout_rows_used != 0`
的训练 summary。

## 远程进程

- actor runner PID：`3854560`；
- actor guardian PID：`3854696`；
- v13 自动续跑 PID：`3910966`；
- 三者均为 PPID 1 / 独立 session，记录时均存活；
- GPU：RTX 4090 D，UUID `GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`。

actor 输出：

```text
/home/user/etsf_robotwin2_actor_execute5_vs_execute50_full400_20260831_v2_stable_roster
```

v13 自动续跑根：

```text
/home/user/etsf_robotwin2_v13_crossbody_selected_20260831
```

权威等待状态：

```text
/home/user/etsf_robotwin2_v13_crossbody_selected_20260831/continuation_state.json
status = waiting_for_authenticated_200_pair_actor_protocol_report
```

旧 v12 等待进程 PID `3891758` 已在仍未产生 protocol、collection 或 checkpoint 时正常终止；旧目录
保留作审计，没有删除。actor runner 与 guardian 未被停止或重启。

## 自动后续链

完整 200 对和 guardian completion 通过 SHA 链认证后，v13 watcher 将：

1. 回放 200 行 outcomes 与预注册层级选择规则，tie 时 fail closed；
2. 生成不可变 `actor_execution_protocol.json` 与选择 receipt；
3. 按所选 execute5/execute50 协议采集五本体 primary C，共 2000 decisions / 8000 branches；
4. 采集并物化 source-only supplement B；
5. 训练五个 held-out-body folds，每折五成员、每成员 3000 steps 的 v13；
6. 在同一 body-condition-seed schedule 上执行 N=1/4/8，共 1000 triplets / 3000 rollouts；
7. 输出二值成功率、阶段进度、配对差值、requested-seed cluster CI 与 exact McNemar。

服务器记录时尚余约 50 GiB；现有同格式 1572 个 primary 文件仅约 33 MiB，因此当前分支数据、
checkpoint 与 JSON outcome 规模不会接近剩余空间，但仍应在阶段切换时复查磁盘。

## 仍未完成

- v13 相对 empirical/v12 的完整同预算离线消融；
- matched VLA-ATTC-style Relative Action Critic 和 WCM-style future-latent baseline；
- 真正 candidate oracle regret 需要对 query0 的八个候选各自执行 one-deviation branch，不能从
  N1/N4/N8 三条已经分流的闭环 rollout 伪造；
- 多任务/双留出实验尚未开始，当前最终结论范围只能是单任务 critic LOBO 跨本体迁移。

## 08:02 CST 续跑指针更新

统一 N1/N4/N8 报告 materializer 与真实 query0 oracle 合同已在提交
`717ddb9e34996b3c2bdd63795f25a771a6988ea0` 落地。等待中的旧 v13 watcher 在尚未产生训练/采集
产物时被提交 `717ddb9` 的 watcher 正常替换；actor runner 与 guardian 仍为原 PID，没有损失
rollout。

当前权威自动续跑：

```text
PID:       3923793
code:      /home/user/etsf_robotwin2_v13_protocol_code_717ddb9
run root:  /home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831
state:     /home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831/continuation_state.json
```

该版本会在 nested 3000 rollouts 完成后自动物化 `crossbody_final_report.json`。没有独立八候选
truth 时仍完整报告标准迁移指标，但严格写入 `oracle_evidence_sufficient=false`、
`oracle_regret=null`；不会把三条已分流策略 rollout 冒充 counterfactual oracle。

## 08:09 CST 效果与 RAC 基线更新

远端 actor 协议比较继续由原 runner/guardian 执行；本次读取到 54 个完整 pair 文件：

| 已完成 cell | execute5 | execute50 | 局部差值 |
| --- | ---: | ---: | ---: |
| aloha-agilex / clean（20） | 75.0% | 50.0% | +25.0pp |
| aloha-agilex / randomized（20） | 50.0% | 50.0% | 0.0pp |
| arx-x5 / clean（14） | 57.1% | 64.3% | -7.1pp |
| 当前 pooled（54） | 61.1% | 53.7% | +7.4pp |

这是未完成且按 roster 顺序逐 cell 产生的 actor 执行时域比较，既不能作为最终协议选择，也不是
共享头增益。权威 `progress.json` 在并发读取时为 `53/200`，随后第 54 个 pair 原子落盘，属于正常
运行中的瞬时差异。

matched VLA-ATTC-style Relative Action Critic 在 08:09 时已实现单折五成员 LOBO 训练器和 N4/N8
全 pair soft-Copeland 运行时适配器；当时定向联合回归 163 项通过、尚未接入 nested runner。其后
闭环接线状态见下一节；当前仍无 RAC 闭环成功率结果，不能用代码测试代替效果数字。

## 08:32 CST RAC 正式闭环接线

RAC 已增加五折自动 supervisor 和 nested `--critic-kind rac` 路径。supervisor 支持：

- 主 v13 完成收据与最终报告未冻结前不打开 branch payload、不竞争 GPU；
- supplement binding 尚未生成时只读等待，生成后验证 actor protocol 与 logical SHA，再一次性冻结
  file SHA；
- 每折五成员、每成员 3000 steps，create-once attempt、失败保留、原子晋升和中断恢复；
- 五本体 N4/N8 使用全 pair soft-Copeland，并把 pair matrix、member score、均值/方差、epistemic
  LCB 和最终选择全部写入可重放 rank receipt。

相关 RAC、supervisor、nested、最终报告与 postformal watcher 定向回归共 71 项通过。该队列仍须部署
到 4090，且必须排在主 v13 最终成功收据之后；因此这里仍只记录执行能力，不记录尚未产生的 RAC
成功率。

## 08:36 CST matched WCM-style 对照

已实现近等容量 future-latent 基线：221,558 参数，对应 v13 的 223,287 参数（0.9923x）。输入为
causal state/history 与 candidate action，输出有限时域 canonical consequence/effect latent、
success/value/event/object effect；目标包括 latent MSE、proper loss、SIGReg 与 variance/covariance。
runtime 明确拒绝候选 outcome 字段，source-only normalization/checkpoint selection 保持 held-out
payload zero-open。相关 WCM/共享头/nested 定向回归本次复验 162 项通过。

当前只完成模型、单折五成员 trainer、N4/N8 scorer 和文档，尚未部署远端五折 supervisor 或闭环
runner。因此它是已实现但未训练的强对照，不能用参数量、单元测试或 source validation 代替任务
成功率。

## 08:39 CST v13 真实 materializer 阻断修复

静态审计发现生产 `_npz_rows()` 原先只把 condition 编入 `logical_group`，没有生成独立
`row["condition"]`；v13 `source_causal_stratum_proper_weights()` 会读取该字段，所以真实五折训练会
在进入参数更新前 fail closed。现已把 manifest-visible condition 附到 source row，并在
`TransitionDataset` 张量化前剥离，确保 condition 只用于 source 因果分层、不会成为模型输入。

新增回归从真实 fixture NPZ 经 materializer 直接调用 v13 balance；共享头/WCM/watcher 相关 164 项、
RAC/nested/final-report 相关 58 项均通过。旧远端 v13 watcher 仍处于 actor 协议等待阶段、尚未产生
primary/supplement/checkpoint，因此必须用包含此修复的新只读代码快照替换等待 watcher，actor
runner/guardian 不需要重启。
