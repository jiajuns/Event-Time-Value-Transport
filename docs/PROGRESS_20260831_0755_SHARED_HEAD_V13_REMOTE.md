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
