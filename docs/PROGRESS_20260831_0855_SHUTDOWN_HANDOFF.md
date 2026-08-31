# 关机交接记录（2026-08-31 08:59 CST，最终复核）

## 远端 4090 实况

RoboTwin2 actor 执行协议比较仍在运行：

```text
完成：       96 / 200 pairs，192 / 400 rollouts
最后一对：   franka / clean / seed 2026104022
pair failure:   0
method failure: 0
```

以下进程均为 PPID 1、独立 session；本机关闭不会终止：

| 阶段 | PID | 当前状态 |
| --- | ---: | --- |
| actor execute5/execute50 runner | 3854560 | running |
| actor guardian | 3854696 | running |
| 修复版 v13 主链 | 3984587 | waiting for authenticated 200-pair report |
| RAC 五折 LOBO 队列 | 3975210 | waiting for primary protocol binding |
| RAC N1/N4/N8 后半链 | 3988427 | waiting for authenticated RAC five-fold completion |

主 v13 输出根：

```text
/home/user/etsf_robotwin2_v13_crossbody_selected_r2_20260831
```

RAC 五折输出根：

```text
/home/user/etsf_robotwin2_rac_lobo_after_v13_20260831
```

RAC 闭环输出根：

```text
/home/user/etsf_robotwin2_rac_nested_success_after_lobo_20260831
```

## 已冻结代码

- v13 主链代码：提交 `81d4724ba221d3685a75862f8b8e098b9772608b`，只读目录
  `/home/user/etsf_robotwin2_v13_protocol_code_81d4724`；包含真实 NPZ row 的 condition 元数据修复；
- RAC 五折代码：提交 `a25514f73c0d31bc454767108e96ed5798d9f543`，只读目录
  `/home/user/etsf_robotwin2_rac_code_a25514f`；
- RAC 后半闭环代码：提交 `fc7d897e8ac457db2128bfa1e403084dfa80e795`，只读目录
  `/home/user/etsf_robotwin2_rac_nested_code_fc7d897`。

RAC 后半 watcher 首次用 RoboTwin2 父解释器启动时因该环境没有 `h5py`，在模块导入阶段退出；没有
创建实验 state、authority、failure receipt 或 rollout。随后改用 ETSF_RoboTwin 作为 watcher 父解释器，
真正 nested 子进程仍固定使用 RoboTwin2。重挂 PID `3988427` 已生成健康 waiting state。

## 自动顺序

```text
200-pair actor protocol
  → primary 2000 decisions / 8000 branches
  → supplement
  → v13 五折共享头（5 bodies × 5 members × 3000 steps）
  → v13 N1/N4/N8（1000 triplets / 3000 rollouts）+ 最终报告
  → RAC 五折（同 5×5×3000）
  → RAC N1/N4/N8 + 独立最终报告
```

三个 watcher 都先验证上游完成权威再继续；RAC 不会在 v13 阶段切换的瞬时空档抢 GPU。

08:58 CST 最终进程复核：上述五个 PID 均存活且 `PPID=1`；actor guardian 为
`monitoring`、`restart_count=0`。因此本机断网或关机不会终止这些远端任务。

## 当前效果边界

- 还没有完整共享头或 RAC 的跨本体 `Delta SR`；当前不能宣称共享头已经提高成功率；
- 先前 54 对时 actor execute5 相对 execute50 的未配平局部趋势为 `+7.4pp`，只用于监控；
- 正式结论必须等待五本体两条件完整 N1/N4/N8、阶段进度、cluster CI 与 exact McNemar。

## WCM-style 对照暂停点

已提交的 matched WCM-style 主体包含 221,558 参数模型、单折五成员 trainer、N4/N8 scorer、文档与
测试；提交历史已在当前 main 中。关机前主动中断了仍在本机编辑的 WCM 正式编排子任务，避免留下
并发写入。未完成、未提交、未部署的工作树仅包括：

```text
M  scripts/train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py
?? scripts/watch_robotwin2_v13_rac_to_wcm_lobo_training_v1.py
```

前者正在增加 summary trainer SHA / logical SHA；后者是等待 v13+RAC 完成后再训练 WCM 五折的
supervisor 草稿。恢复时应先审查和补测试，不能把它们直接当作已完成队列启动。

用户原有未提交文件仍保持原状，不属于本轮提交：OpenVLA rollout/shadow 两个脚本、`artifacts/` 和
SmolVLA r17 launcher。
