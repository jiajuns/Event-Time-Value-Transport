# 2026-08-31 动作协议实验失败审计与 v2 恢复计划

## 1. 当前结论

`execute-5` 与 `execute-50` 的首轮比较没有完成，不能据此选择正式部署协议。
远程 v1 在完成 26/200 个配对后，于下面这个场景初始化阶段失败：

- body：`aloha-agilex`
- condition：`randomized`
- requested seed：`2026101006`
- error：`UnStableError: Objects is unstable ... 001_bottle`
- method start/result：两种方法均为 0；失败发生在 actor 推理和动作执行之前
- immutable pair failure receipt：
  `pair_failures/aloha-agilex__randomized__seed_2026101006.json`

守护器看到正式 failure receipt 后按 fail-closed 合同退出，没有重启、换 seed、删除文件或
修改已完成结果。旧输出根保持只读审计用途：

`/home/user/etsf_robotwin2_actor_execute5_vs_execute50_full400_20260831_v1_endpose_eventv2`

## 2. 旧 26 对只能作为失败审计

停止时的描述性统计如下，不能用于冻结协议，也不能并入 v2 正式报告：

| 方法 | 成功 | 阶段进度均值 | 目标距离改善均值 |
| --- | ---: | ---: | ---: |
| execute-5/replan | 10/26 | 0.5096 | -0.1010 m |
| execute-50/native | 12/26 | 0.6250 | +0.0109 m |

discordance 为 `execute50_only=7`、`execute5_only=5`、共同成功 5、共同失败 9。样本仅来自
第一个本体且不完整，因此既不是五本体动作协议结论，也不是共享事件头或跨本体效果。

## 3. v2 的结果盲稳定 seed roster

RoboTwin 官方评测在 `setup_demo` 抛出 `UnStableError` 时会跳过该 seed。v2 采用相同原则，
但把 seed 选择提前到任何 actor 推理和 outcome 之前：

1. 使用全新、未检视的候选区间，从 `2026104000` 开始；不复用 v1 或非正式探针 seed。
2. 固定 body、condition 和 candidate seed 的遍历顺序。
3. 每次只执行 fresh scene setup、对象发现、reset snapshot 和一次 canonicalization scene step；
   不加载 actor，不生成候选，不执行动作，不读取成功、事件或候选结果。
4. 一个 candidate seed 只有在 5 bodies × 2 conditions 全部稳定时才能进入 roster。
5. 按候选顺序选择前 20 个共同稳定 seed，先生成 create-once roster JSON，并绑定文件 SHA 和
   logical SHA；正式 runner 禁止边执行边补 seed。
6. v2 使用新输出根完整重跑 200 pairs/400 rollouts；v1 的 26 对不复用。

## 4. 与最终共享头的关系

该比较只决定 actor action chunk 的正式执行步长。完成后才执行以下主链：

1. 按胜出协议采集五本体 8,000 个真实 counterfactual branches 和 scripted-root supplement；
2. 训练五折 leave-one-body-out 共享事件头；
3. 联合评估下一事件、持续时间、成功/恢复、对象变化和不确定性；
4. 在 held-out body 上比较 actor-only 与 N=4/N=8 critic reranking 的配对任务成功率和阶段进度。

当前尚无可成立的“共享头提高跨本体成功率”结论。

## 5. 代码与远程状态

- GitHub 已推送提交：`cd761af`
- v1 runner code：
  `/home/user/etsf_robotwin2_fivebody_terminal_v22_code_77a6970`
- guardian code：
  `/home/user/etsf_robotwin2_fivebody_terminal_v23_code_cd761af`
- v1 runner PID `3814405`：已退出
- guardian PID `3841435`：已记录 failure 后退出
- method failure 数：0
- pair failure 数：1
