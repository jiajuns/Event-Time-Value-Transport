# Schema-v5 反事实微调安全 launcher

入口：`scripts/launch_openvla_etsf_counterfactual_v5.py`。该脚本应使用远端 RoboTwin venv 的 Python 执行；它不会加载 OpenVLA actor，只训练已采集 hidden/action 上的小型事件世界模型。

默认正式命令：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/launch_openvla_etsf_counterfactual_v5.py
```

首次使用建议先运行相同命令并追加 `--dry-run`。dry-run 完成全部 CPU 前置审计并打印最终 trainer argv，不创建输出、不检查/占用 GPU，也不启动训练。

## 固定默认路径

- schema-v5 train100：`/home/user/etsf_openvla_event_branches_v5_train100_20260827`
- structured factual 三成员：`/home/user/etsf_openvla_structured_event_world_model_move_can_pot_sealed_schema3_20260827`
- event spec：`/home/user/etsf_stage2_run_20260825/event_spec.json`
- 默认 counterfactual 输出：`/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry1_20260827`

旧目录 `...move_can_pot_20260827` 保留首次启动失败的审计证据，不得覆盖或复用；
`retry1` 已完成但组内排序较弱，保留为只读证据；排序修复必须显式指定全新
`/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry2_rank_20260827`
目录，不能覆盖或 resume 旧输出。默认值仍保留 retry1，避免 fresh50/progress 审计在
retry2 尚未完成时误引用新目录。

retry2 命令（同步新 trainer/launcher 后再执行）：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/launch_openvla_etsf_counterfactual_v5.py \
  --output /home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry2_rank_20260827
```

retry1 validation 的 baseline/oracle 为 `3/15`、`4/15`，说明 15 组里理论上只有一组
可以改善；三个成员 top-1 为 `0/1/1`。global candidate AUC 约 `0.81`，但组内 pairwise
accuracy 仅约 `0.19–0.25`，表明绝对 success logit 主要识别了场景难度，没有可靠识别
同一状态下的动作差异。ensemble 的 14 次非 baseline proposal 中 helpful 1、harmful 3，
因此 guard disabled 是正确的安全结果，不应降低 guard 门槛来掩盖排序失败。

retry2 保留全部 event/time/object/latent 监督，并新增两个候选级损失：组内中心化 score
去除 scene difficulty，以及只比较 success-changing candidate 与 deterministic fallback
的 baseline contrast。member checkpoint 只按 validation 的组内 pairwise Wilson 90% LCB、
top-1 uplift、event loss、total loss依次选点；fresh50/sealed 标签仍不加载。

实际 `retry2_rank` 因 checkpoint 路径与 collector 的 `openvla` 策略别名不一致，在零训练
step 时 fail-closed；该目录只保留失败审计。修复别名规范化后，正式诊断输出改为：

```text
/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry2b_rank_20260827
```

`retry2b_rank` 已完成三个成员，但 15 组 validation 上仍为 14 次非 baseline proposal、
helpful 0、harmful 3，unguarded success 为 0/15，故 guard 继续 disabled。全 100 个开发组
实际有 baseline success 12、oracle success 32、20 个 oracle-headroom group；下一步应使用
固定的 group-level 5-fold OOF 利用全部开发证据，不能继续针对这 15 组调参，也不能降低 guard。

collector 未完成时默认等待最多四小时，每 30 秒复查。只有 `status=complete`、schema 5、恰好 100 个唯一 train seed/组、4 候选、完整语言契约且 100 个 HDF5 identity attrs 均一致时才继续。collector resolved seeds 还必须恰好等于 factual contract 的 100 个 train seeds，并与 factual validation/sealed seeds 无交集；launcher 不打开 success、event 或轨迹标签 dataset。

collector 完成后，launcher 还会等待固定目录 `seed_20260827/28/29` 的三个 factual member 全部出现并达到 `training_complete`（而不是在第一个 seed 训练期间误判失败）；缺成员或仍在训练会继续轮询，额外 seed summary 则按污染立即拒绝。三者必须未评 sealed test、共享同一数据/split/event-spec contract，且 summary 和 best checkpoint 均为 `structured_events=True`。每个 summary/checkpoint 的 `contract.training_seed` 必须等于其目录 seed；成员间比较共享契约时只排除该预期不同字段，其余字段必须完全相同。脚本逐个验证 best checkpoint 与 summary contract/score 和 SHA，然后按最小 `best_validation_selection_score` 选择 factual 初始化，并在 `launch_audit.json` 记录完整候选表、选择规则与 SHA。`--wait-timeout-seconds` 对 collector 和 factual 两个阶段分别提供等待预算。

正式 argv 固定为三个 seeds `20260827/28/29`、CUDA、bf16，调用 `train_openvla_etsf_counterfactual.py`。正式启动还会验证 GPU 名称包含 `4090`。考虑 factual 最后一个 summary 已原子写出、但训练进程尚未释放 CUDA 的短暂竞态，launcher 会再用 `nvidia-smi` 等待 GPU 0 上除自身外的 compute PID 清空；默认最多四小时、每 30 秒检查，可用 `--gpu-wait-timeout-seconds` 与 `--gpu-poll-seconds` 调整。无法解析进程列表或超时均安全拒绝，等待结果写入 launch audit；`--dry-run` 不执行任何 CUDA/nvidia-smi 探测。

三个模型完成后，scoring 不再使用未经验证的固定 `.25/.05` 权重。trainer 只在 validation
上比较预注册的 7 个候选：`success_only`、`success_distance`、`progress_light`、
`progress`、`progress_clock`、`full_light`、`full`；不允许连续或自适应扩大搜索。
其中含距离项的候选固定使用 launcher 显式传入的 `0.02`，不会再用同一 validation 连续调权。
所有候选的 proposal 数、coverage、帮助/伤害数、paired delta、90% LCB 和 unguarded
success rate 都写入 `scoring_selection`。按预注册的 LCB→policy success→mean delta→
保守 grid order 规则冻结一个 scoring，之后才在固定至多 3×3 的 gain/uncertainty 分位数
网格调 guard。正式 guard 必须同时满足：至少 10 个 proposal、coverage ≥10%、paired
success delta 的 90% LCB ≥0、harmful rate ≤10%；否则 `guard.enabled=false`。
manifest 与 aggregate checkpoint 镜像 `scoring`、完整 `scoring_selection` 和 `guard`。

当前 counterfactual trainer 没有 resume 接口，因此 launcher 不伪造恢复能力：

- 输出不存在或为空：允许首次启动，并以 `launch.lock` 防并发；
- 输出具有完整、SHA 一致、相同数据/初始化的三成员 ensemble：安全跳过；
- 任何部分、失败、不同数据或不同 checkpoint 输出：拒绝启动，必须换一个明确的新输出目录。

该 launcher 只接收 v5 train100，不接触原 official test 或 fresh confirmation 数据。
