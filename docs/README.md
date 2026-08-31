# docs

这里只保留当前仍有效的四份文档：

1. `CURRENT_STATUS.md`：远程采集、训练队列和权威状态路径；以后原地更新。
2. `ETSF_ROBOTWIN2_SHARED_HEAD_V13_CAUSAL_BALANCE_ZH.md`：当前共享头模型与五折训练定义。
3. `ROBOTWIN2_CROSS_EMBODIMENT_EVALUATION_PROTOCOL_V3_ZH.md`：当前正式评测指标与结论边界。
4. `README.md`：本目录入口。

旧 OpenVLA/SmolVLA 路线、Stage0–Stage3、v6–v12 设计、重复运行说明和时间戳进度快照已经从当前工作树删除。需要追溯时使用 Git 历史，不再把历史文件放回 `docs/`。

维护规则：

- 状态变化只更新 `CURRENT_STATUS.md`；
- 模型发生实质变化才新增 v14，并删除被取代的 v13；
- 测评口径发生实质变化才新增 v4，并删除被取代的 v3；
- 日志、测试输出、临时分析和 checkpoint 不进入 `docs/`。
