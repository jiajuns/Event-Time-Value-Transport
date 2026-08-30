# 共享头 v12 与 actor 执行协议 v2 断点（2026-08-31 06:57 CST）

## 1. 当前结论

- 现在不能声称共享头已经提高跨本体成功率。旧的不完整 26 对实验中，execute5 为
  `10/26`、execute50 为 `12/26`，但该运行被未预筛的 `UnStableError` 中断，只能视为
  描述性趋势，不能用于协议选择或论文主张。
- 新的共同稳定 seed 协议已经完成，正式 execute5 与 execute50 对比已经在远程 RTX 4090
  启动。最终执行时域必须由完整 200 对结果决定，不能提前根据旧 26 对固化。
- 共享头已升级为 v12 的下一事件—持续时间竞争风险模型并完成数值加固；真实五折 LOBO
  尚未用 v12 重训，旧 v11 checkpoint 不能改名复用。

## 2. 已推送代码

- `7b6ade9`：正式五本体 LOBO N=1/N=4/N=8 指标、cluster bootstrap CI 与 McNemar。
- `e2fc090`：actor 协议从冻结共同稳定 seed 名单恢复。
- `7b06289`：共享头 v12 联合 `p(next event) p(D | next event)`。
- `c5c1c6b`：彻底隔离旧 duration loss、严格二值 mask、total duration uncertainty。
- `02f37b4`：稳定名单完成后无人值守启动 runner 并与 guardian 权威握手。

本地定向回归：

- 共享头、离线 LOBO、插件及 N=1/4/8 链：`162 passed`；
- roster、runner、guardian 与 bootstrap 链：`36 passed`；
- `py_compile` 与 `git diff --check` 通过。

## 3. 共享头 v12 的实质修改

v12 不再按 `current_event_id` 硬选 duration 分量，而是建模：

```text
p(e_next, D | e_t, canonical action effect)
= p(e_next | e_t, action effect)
  p(D | e_next, e_t, action effect)
```

- observed boundary：下一事件 CE 加真实目的事件分量的 duration NLL；
- right-censored boundary：`-log sum_e p(e) S_e(D_censor)`；
- duration total uncertainty：成员内 next-event 分量内方差、分量间方差与五成员 epistemic
  方差按 total variance law 合并；
- success、failure、条件恢复、联合恢复、对象变化和各事件熵均为显式输出；
- 不输入 robot ID、不建立本体私有 head，五折 LOBO 的留出本体 payload、统计量、校准和
  checkpoint 选择保持 zero-open。

已知限制：v12 延续 `Y=log1p(D)` Gaussian，理论上会给 `D<0` 分配极小概率，而部署点预测
截为非负。该限制已在设计文档披露，不能在同一模型版本中静默更换 likelihood。

## 4. 远程共同稳定 seed 名单

- 主机：`user@100.115.128.14`；
- GPU：`GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`，RTX 4090 D；
- 名单：
  `/home/user/etsf_robotwin2_actor_protocol_v2_stable_seed_roster_20260831/stable_seed_roster.json`；
- 文件 SHA-256：
  `46d751c5a1d04602ac7296ba51c63e913fa02fe82e5aad87edeb0b6963387a60`；
- logical SHA-256：
  `8fb3582eb55f71a416bdd1a22132eb5d54aaa68487c811c6244afcd635d5f348`；
- 候选尝试数：28；第一批满足五本体乘两条件全部稳定的 20 个 seed：

```text
2026104000 2026104001 2026104002 2026104005 2026104006
2026104007 2026104009 2026104011 2026104012 2026104013
2026104015 2026104017 2026104018 2026104019 2026104020
2026104022 2026104024 2026104025 2026104026 2026104027
```

筛选只执行 setup/close；actor inference、task action、label/outcome read 均为 0。

## 5. 远程正式 actor 对比

- immutable runner code：
  `/home/user/etsf_robotwin2_fivebody_terminal_v25_code_e2fc090`；
- runner SHA-256：
  `3be6cb7248be389a223b875efe7ff8a3dd55fc0c21ae243ddb8bce82f8207e66`；
- guardian SHA-256：
  `2dacf3eceb4efc19e537512c8e536b559f9ced2a9296e0adb380b6d4ce947dd4`；
- bootstrap code：
  `/home/user/etsf_robotwin2_actor_bootstrap_v26_code_02f37b4`；
- bootstrap complete logical SHA-256：
  `fc233981e3021f66123672b2275d7d5cfa2d649c9c430f2a10539f0b8bb22580`；
- runner PID：`3854560`；guardian PID：`3854696`；两者 PPID 均为 1；
- 正式输出：
  `/home/user/etsf_robotwin2_actor_execute5_vs_execute50_full400_20260831_v2_stable_roster`；
- guardian 状态：
  `/home/user/etsf_robotwin2_actor_execute5_vs_execute50_guardian_v2_stable_roster`；
- 计划：五本体乘两条件乘 20 seed，共 200 对、400 rollout；
- 记录时：guardian `monitoring`、restart 0、completed pairs 0，第一对 execute5 正在运行；
  GPU 约 64% utilization、6.3/24.6 GiB。

bootstrap 已完成 roster、runner binding、guardian plan 和 `monitoring` 状态认证后退出；关闭本机
不会中断 runner 或 guardian。

## 6. 数据与协议链当前阻塞

旧 execute5 正式 C 数据根停在 1572/2000 decisions：

```text
/home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip
```

不能直接用它完成 v12 最终训练，因为执行时域尚未由正式对比选定。当前四个本地 partial
文件只完成了采集侧的 40x5 / 4x50 动态分配，尚不构成可部署链：

- primary watcher 的 `run_root` 与 HOME-relative 路径解析会形成重复路径；
- trainer、LOBO watcher 和 postformal watcher 仍存在 stride=5、40 queries、`dt=5/15`
  硬编码；
- primary v3、supplement v4 与下游 v2/v3 schema 尚未贯通；
- supplement binding 尚未逐层绑定同一 protocol file SHA；
- postformal watcher 没有给新 supplement collector 传协议参数，nested runner 仍固定 execute5。

因此这些 partial 保持未提交、未部署。正式 actor 对比完成后，才把胜出的单一冻结协议文件
贯通到 C/B collector、binding、trainer、LOBO aggregate 和 N=1/4/8 runner。

## 7. 下一步完成标准

1. 完成 execute5/execute50 的 200 对正式比较，报告成功率、阶段进度、95% cluster CI 和
   McNemar，并冻结胜出执行协议；
2. 按冻结协议补齐或重采完整 C 数据，并采 e12/e3/e4 source-train-only supplement；
3. 从头训练五折 LOBO x 五成员 v12，报告 held-out next-event、duration、success/recovery、
   object effect 与 uncertainty 指标；
4. 用同一冻结 actor 和每折对应 shared head 完成五本体 N=1/4/8 配对闭环；
5. 只有五个留出本体的宏平均 `Delta SR`/`Delta Stage` 及置信区间支持改善，才声称实现了
   可迁移且提高任务成功率的事件世界模型。
