# RoboTwin2 五本体正式配对成功率执行器

`scripts/run_robotwin2_five_body_paired_success_v1.py` 是真实 simulator 执行器，不是离线指标
汇总器。它执行预注册的完整协议：

```text
5 held-out bodies × 2 conditions × 100 fixed seeds
= 1000 paired trials
= 2000 real rollouts
```

每个 `(heldout body, condition, requested seed)` 的两个方法均从独立但相同的 deterministic reset
开始。偶数 seed ordinal 按 actor→ETSF 执行，奇数按 ETSF→actor 执行。每次策略查询都由同一个冻结
EE16 actor 按固定 flow-noise 规则生成四个有序候选：

- `actor_baseline` 固定执行 `candidate_index=0`；
- `etsf_best_of_4` 使用该 body 对应的 LOBO 五成员共享头；每个成员先在当前四候选内对 raw rank
  logit 做 population z-score，再对恰好五成员等权平均后选择；
- 分数精确相同时，选择最低 candidate index；
- 两个方法的初始 reset 和第一次四候选集合必须一致；动作分叉后，各自在自己的真实闭环状态继续查询并
  执行动作，不使用 simulator lookahead 或候选 rollout outcome。

执行器会先冻结一次初始承诺，然后在每个方法执行第一步动作前重新核验。reset identity 覆盖全部已发现
tracked object poses、双臂 articulation qpos/qvel（以及运行时公开的 qacc/qf）、全部 active-joint drive
targets、commanded arm state、EE16 和 simulator/task counters；候选承诺绑定有序 `[4,H>=5,16]`
数组 SHA。每个 pair 在模拟前写一次性 attempt receipt；若运行时/check-success/承诺校验异常，写不可变
failure receipt 并停止，后续启动不会静默重跑该 pair。动作本身的 infeasible 异常仍作为真实 rollout
失败记录。

共享头在线输入沿用训练时的解析式跨本体接口：27D canonical event state、14D SE(3)+gripper
动作效果和秒制 `dt`。共享头只对候选打分；二值成功来自 RoboTwin `eval_success/check_success`，事件
阶段进度来自真实轨迹的 `e0→e12→e3→e4→eK` 最大到达阶段。脚本不读取官方 expert archive，
不读取受保护的内部 HDF/label，也不进行任何训练或参数更新。

## 输入绑定

正式执行必须提供：

- 已训练并冻结的统一 state16/action16 EE actor checkpoint；
- actor 对应的本地 VLM metadata；
- 官方 RoboTwin2 simulator tree；
- frozen analytic event spec，文件 SHA 为
  `4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee`；
- SHA 为 `a4e59f647c520609313e1c9aca03dbb3f770504e0383c66bb619dca94b4c6827`
  的 corrected v2 正式五本体指标预注册 JSON；
- 五个 LOBO fold 目录，每个目录包含 `training_summary.json` 和五个 hash 匹配的 member checkpoint。

每个 `--lobo-fold` 使用 `heldout-body=/absolute/path`，必须恰好提供五次。执行器会核对 summary 的
held-out identity、无 held-out 监督、单共享 action stem、五成员编号和所有 checkpoint SHA，然后仅
加载对应 held-out body 的 ensemble。
五个 checkpoint 必须来自同一个 source-only 共同评估 step，该 step 已用与在线完全相同的标准化
ensemble 在 source validation 上联合选择；summary、checkpoint 和当前 trainer 实现 SHA 必须一致。

## 远程 4090 命令

RoboTwin/CuRobo 必须使用已验证的 RoboTwin2 Python，LeRobot 0.4.4 通过显式 PYTHONPATH 注入；不要
直接用带 user-site Torch 2.10 的环境启动 simulator。

```bash
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/user/etsf_stage0/lerobot/src:/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages:/home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/lib/python3.10/site-packages:/home/user/etsf_stage0/RoboTwin:/home/user/etsf_stage0/RoboTwin/envs/curobo/src

/home/user/anaconda3/envs/RoboTwin2/bin/python \
  scripts/run_robotwin2_five_body_paired_success_v1.py \
  --actor-checkpoint /ABS/FROZEN_EE16_ACTOR \
  --vlm-metadata-path /home/user/etsf_smolvla_models/smolvla_base_c83c3163 \
  --robotwin-root /home/user/etsf_stage0/RoboTwin \
  --event-spec /home/user/etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json \
  --preregistration /home/user/public_benchmark_receipts/robotwin2_move_can_pot_5emb_metrics_preregistration_20260830_v2.json \
  --lobo-fold aloha-agilex=/ABS/LOBO_ALOHA \
  --lobo-fold arx-x5=/ABS/LOBO_ARX \
  --lobo-fold franka=/ABS/LOBO_FRANKA \
  --lobo-fold piper=/ABS/LOBO_PIPER \
  --lobo-fold ur5=/ABS/LOBO_UR5 \
  --output /ABS/NEW/paired_success_run_v1 \
  --action-exec-steps 5 \
  --max-steps 200 \
  --fps 15
```

`fps=15` 与统一 EE16 actor 的训练数据时间基准绑定；因此 planned first chunk 的五个 token 在共享头
输入中固定为 `dt=5/15` 秒。该 `dt` 在候选执行前构造，不使用事后 `first_executed`。另一方面，
事件持续/静止判断和详细执行审计使用 collector 的 SAPIEN physical-step clock，逐动作记录真实
`physical_sim_seconds`；planned actor-token 时间与执行后的 simulator 物理时间不会混写。

正式长任务应由服务器端 `setsid`/`nohup` 包装，stdout/stderr 写入服务器日志。输出目录可恢复：每完成
一个 pair 就原子写入只读 `pairs/<identity>.json`。重启时不仅校验顶层 SHA，还会逐 query 重算
`raw[5,4] → decision内标准化 → ensemble[4] → selected index`，并强制 baseline 始终执行 candidate 0；
任何旧格式或无法复算的 pair 都会停止而不是被静默接纳。流程不换 seed、不删除失败 pair，也不根据
outcome 提前停止。

## 输出与现有 evaluator

每个详细 pair 记录包含：

- 两方法二值成功、阶段进度和 discordance；
- 初始 reset identity 与第一次 candidate-set SHA；
- 每次 query 的候选集合 SHA、selected candidate index 和实际执行步数；
- ETSF 的五成员 raw rank logits、每成员候选 mean/std、标准化 ensemble rank、
  success/post-event/next-event 预测。

全部 1000 pairs 完成后才冻结发布 v3 `paired_outcomes.json` 和
`paired_execution_completion_receipt.json`。每个 outcome row 绑定对应详细 pair SHA，顶层绑定有序
1000 个 pair SHA 摘要、execution-contract logical/file SHA；completion receipt 再绑定五折 checkpoint、
标准化聚合合同、outcome document/file SHA。evaluator 仍只读取严格结果字段和这些 SHA，不打开
checkpoint/轨迹。执行器最后会打印文件 SHA 和可直接复制的命令：

```bash
python3 scripts/evaluate_robotwin2_cross_embodiment_paired_success_v1.py \
  --input /ABS/RUN/paired_outcomes.json \
  --input-file-sha256 PRINTED_FILE_SHA256 \
  --output /ABS/RUN/paired_success_report.json
```

汇总结果包括逐 body×condition、逐 body macro、逐 condition macro 和 global macro 的 SR、配对
`ΔSR`、95% CI、actor-only/ETSF-only discordance、exact McNemar，以及阶段进度指标。
