# 模型与远程队列交接（2026-08-31 10:47 CST）

## 已完成代码

最新 main：`73ae0bb37d1726c38f7ceaff075622d29c6664f7`。

- v13 next-event competing risk 排除 `next_event == current_event`；无事件边界只通过右删失表达；
- 单标量 utility 改为成功概率主项与有界的失败条件事件后果残差；成功概率优势大于
  `0.05` 时不会被阶段、时长、对象效果或不确定性项逆转；
- WCM-style 参数匹配对照已接入五本体 LOBO、N1/N4/N8、checkpoint/load authority、
  候选 roster 与五成员 mean/std/LCB 的签名回放；
- WCM 五折必须等完整 RAC nested 成功率报告认证后才能查询 GPU；WCM 后半链必须等
  WCM 五折认证完成。

验证：共享头、ETSF/RAC/WCM runtime、nested runner 与 watcher 相关套件 208 项通过；
deferred primary binding 修复的 supervisor 聚焦套件 5 项通过；远端只读代码 AST 复核通过。

## 远程 4090 状态

actor 协议截至 10:47 CST：`184/200 pairs`、`368/400 rollouts`，状态 running。

| 阶段 | PID | 代码/状态 |
| --- | ---: | --- |
| actor runner | 3854560 | running |
| actor guardian | 3854696 | monitoring |
| 修复版 v13 主链 | 4128507 | waiting for authenticated 200-pair report |
| RAC 五折 | 3975210 | waiting for primary protocol binding |
| RAC nested/final | 3988427 | waiting for authenticated RAC five-fold completion |
| WCM 五折 r2 | 4143481 | waiting for v13 and complete RAC final authorities |
| WCM nested/final r2 | 4143482 | waiting for authenticated WCM five-fold completion |

全部存活 watcher 均为 `PPID=1`，不依赖本机连接。

v13 使用只读代码：

```text
/home/user/etsf_robotwin2_protocol_code_19ace4d
commit 19ace4d7623514ed25c084a9f3c467ae52b0944f
```

该提交已经包含共享头的两项模型修复。最新提交 `73ae0bb` 只在此基础上修正 WCM
supervisor 对未来 primary binding 的等待顺序；WCM 两条链使用：

```text
/home/user/etsf_robotwin2_protocol_code_73ae0bb
```

权威 WCM 输出根：

```text
/home/user/etsf_robotwin2_wcm_lobo_after_rac_final_r2_20260831
/home/user/etsf_robotwin2_wcm_nested_success_after_lobo_r2_20260831
```

第一次不带 `r2` 的 WCM 启动因过早要求尚未生成的 primary binding 而 fail-closed；其失败
state/run-exit 被保留用于审计，不应作为实验结果。`r2` 已验证健康，run-exit 尚不存在且日志为空。

## 自动执行顺序

```text
actor 200-pair protocol
  -> primary 2000 decisions / 8000 branches
  -> supplement
  -> 修复版 v13 五折 + v13 N1/N4/N8 final
  -> RAC 五折 + RAC N1/N4/N8 final
  -> WCM 五折 r2 + WCM N1/N4/N8 final r2
```

## 接管时读取的权威状态

```text
/home/user/etsf_robotwin2_actor_execute5_vs_execute50_full400_20260831_v2_stable_roster/progress.json
/home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831/continuation_state.json
/home/user/etsf_robotwin2_rac_lobo_after_v13_20260831/rac_lobo.watcher_state.json
/home/user/etsf_robotwin2_rac_nested_success_after_lobo_20260831/rac_nested.watcher_state.json
/home/user/etsf_robotwin2_wcm_lobo_after_rac_final_r2_20260831/wcm_lobo.watcher_state.json
/home/user/etsf_robotwin2_wcm_nested_success_after_lobo_r2_20260831/wcm_nested.watcher_state.json
```

最终效果仍必须以完整五本体留出本体 `Delta SR`、阶段进度、cluster CI、exact McNemar
以及事件/成功/时长/对象效果的预测与校准指标为准；当前 actor 部分结果不是共享头效果。
