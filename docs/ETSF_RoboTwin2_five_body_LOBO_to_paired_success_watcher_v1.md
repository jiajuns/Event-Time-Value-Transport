# RoboTwin2 五折 LOBO 到正式配对成功率 watcher v1

`scripts/watch_robotwin2_five_body_lobo_to_paired_success_v1.py` 是纯标准库等待器和远程调度器。
它只在上游固定状态文件同时满足以下条件后继续：状态为 `complete`、`run.exit=0`、五折 aggregate
SHA 匹配、五个 `training_summary.json` 满足 outer-LOBO 合同、25 个 member checkpoint 均位于对应
fold 内且文件 SHA 匹配。

等待阶段不导入 Torch、NumPy、RoboTwin 或 simulator，也不创建 GPU reservation。上游完成后，
watcher 冻结 actor tree、VLM metadata tree、analytic event spec/实现、两类不同 preregistration
authority、五折和只读代码身份；连续两次
确认固定 UUID 的 RTX 4090 没有 compute PID，并在 Popen 紧前再次确认，才调用正式 runner。runner
成功产出完整的 `paired_outcomes.json` 后，watcher 释放 GPU reservation，使用 CPU-only evaluator
create-once 生成 `paired_success_report.json`。异常会写真实 `failed` state 和非零 `run.exit`；stdout/
stderr 由服务器 `nohup` 写入固定 watcher log，runner/evaluator 另有追加日志。

## 冻结的部署身份

审计已否决旧 SHA `8b1ff0…` 的 event spec，因为它含 Aloha/ARX 标签
拟合，不能作为正式五本体 analytic event spec。新的 data/label-free analytic contract 已冻结：

- spec 路径固定为
  `/home/user/etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json`，文件 SHA 为
  `4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee`；
- 唯一 event derivation module 为
  `robotwin2_move_can_pot_analytic_event_spec_v1.py`，文件 SHA 为
  `06d0abd896a3bfd7d0ae96141a0753dadcca5d42a42faaa0e2024ed6a7f00d46`；
- 每折 `training_summary.json` 必须同时声明这两个 SHA，runner execution contract 也必须声明同一
  module SHA 与 `training_and_online_event_implementation_identical=true`。

正式 runner SHA 固定为 `068dce8c…`，v2 evaluator SHA 固定为 `01181e82…`；watcher 会在 CUDA
或 simulator 启动前逐文件复核。远程部署必须使用新的只读代码目录，不得把旧 event spec、旧
runner 或已有 dirty artifact 复制成正式授权。

## Materialization v1 与 metrics preregistration v2

两者是不同 authority，watcher 分开记录且禁止互换：

- `robotwin2_move_can_pot_5emb_materialization_a967b852_20260830_v1.json` 只证明既有 11 个公开
  source-slice payload 的完整物化。其 receipt 文件 SHA 为
  `aefd33cd337dbaad5d85e6a7cf5490221cb515fe6bb06462257d279a091f8582`，内部绑定旧 source-slice
  preregistration SHA `75fc9c6e…`；它不冻结本次 paired estimand、统计量或 gate。
- `robotwin2_move_can_pot_5emb_metrics_preregistration_20260830_v2.json` 是本次 prospective paired execution
  与 metrics authority，canonical SHA 为
  `a4e59f647c520609313e1c9aca03dbb3f770504e0383c66bb619dca94b4c6827`。runner 的
  `--preregistration`、outcomes v2 和 report v2 必须绑定此 SHA。

最终 report format 固定为
`etsf_robotwin2_move_can_pot_cross_embodiment_paired_success_report_v2`；materialization v1 的
`75fc9c6e…` 不得再出现在 outcomes/report 的 metrics preregistration 字段中。

## 固定实验边界

其余远程身份已固定为统一 EE16 actor checkpoint、与五本体 branch collector 相同的
`offline_assets/smolvlm2_500m_metadata`、`/home/user/etsf_stage0/RoboTwin`、上述两份分工明确的
authority、
`outer_lobo_<body>` 五个 fold，以及 1000 pairs / 2000 rollouts、5/15 秒首段和 200-step 上限。路径
不通过 CLI 覆盖；CLI 只允许调整 CPU 等待轮询和两次 idle audit 的时间间隔。

本地验证仅执行：

```bash
python3 -m py_compile \
  scripts/watch_robotwin2_five_body_lobo_to_paired_success_v1.py \
  tests/test_watch_robotwin2_five_body_lobo_to_paired_success_v1.py

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_watch_robotwin2_five_body_lobo_to_paired_success_v1.py
```

这些测试只合成 JSON/checkpoint 字节与 GPU 查询返回值，不启动训练、CUDA 或仿真。
