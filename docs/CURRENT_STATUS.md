# 当前状态

> 最近核对：2026-08-31 12:16 CST
>
> 运行服务器：`user@100.115.128.14`，NVIDIA GeForce RTX 4090 D。

## 当前结论

现在仍在采集 v13/RAC/WCM 共用的正式候选分支监督数据，尚未开始共享头参数训练。

| 阶段 | 当前状态 |
| --- | --- |
| actor execute5/execute50 协议 | 已完成：200/200 pairs，400/400 rollouts |
| primary branch 数据 | 采集中：33/2000 decision groups，132/8000 candidate branches |
| v13 五折共享头 | 等待完整 primary binding，0/5 folds |
| RAC 五折 | 等待 primary/v13 authority，0/5 folds |
| WCM 五折 | 等待 v13 与 RAC final authority，0/5 folds |
| 正式 N1/N4/N8 成功率 | 尚未开始 |

采集中的本体为 Piper、ARX-X5、UR5；Aloha-AgileX 与 Franka 尚未开始。核对时 GPU 利用率约 98%、显存约 13.4/24.6 GiB，负载来自三个并行 RoboTwin2 仿真采集进程，不是模型训练。

## 自动执行顺序

```text
primary 2000 decisions / 8000 branches
  -> 冻结 primary_training_binding
  -> supplement
  -> v13 五折 + v13 N1/N4/N8 final
  -> RAC 五折 + RAC N1/N4/N8 final
  -> WCM 五折 + WCM N1/N4/N8 final
```

流水线已经作为 `PPID=1` 的远程 watcher 挂起，不依赖本机连接。数据未完整前不要手动启动 trainer，也不要抢占 4090；完整 binding 生成后会自动进入训练。

## 权威状态路径

```text
/home/user/etsf_robotwin2_actor_execute5_vs_execute50_full400_20260831_v2_stable_roster/progress.json
/home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831/continuation_state.json
/home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831/watcher_state.json
/home/user/etsf_robotwin2_rac_lobo_after_v13_20260831/rac_lobo.watcher_state.json
/home/user/etsf_robotwin2_rac_nested_success_after_lobo_20260831/rac_nested.watcher_state.json
/home/user/etsf_robotwin2_wcm_lobo_after_rac_final_r2_20260831/wcm_lobo.watcher_state.json
/home/user/etsf_robotwin2_wcm_nested_success_after_lobo_r2_20260831/wcm_nested.watcher_state.json
```

当前远程主链使用只读代码：

```text
/home/user/etsf_robotwin2_protocol_code_5932ecd
commit 5932ecdd290c1c4a754bf05ba9854975d6a8e4c2
```

## 结果边界

actor 协议完成只说明执行频率已冻结，不是共享头效果。最终结论必须等五折和闭环全部完成，并以 paired `DeltaSR`、`DeltaStage`、cluster 95% CI、候选 oracle/headroom 以及机制指标为准。

以后直接更新本文件，不再新增带时间戳的 `PROGRESS_*` 文档。实时状态仍以服务器 JSON receipt/report 为最高事实来源。
