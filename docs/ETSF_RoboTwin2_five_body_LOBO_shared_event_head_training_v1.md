# RoboTwin2 五本体 LOBO 共享事件头训练入口 v1

入口：`scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`。

## 目标与结论边界

这个入口把 `move_can_pot` 的五个公开本体 `aloha-agilex / arx-x5 / franka /
piper / ur5` 逐一作为 held-out body，训练同一个 canonical event/effect head。它复用
`MultibodyCanonicalEventWorldModel`，不复制或微调 OpenVLA/SmolVLA；VLA actor 始终冻结。

每折模型内部只有一个共享 body row，并且只有一个真正同义的 task-space action stem：
`dual_ee_se3_gripper_delta_14d_v2`。输入不是各机器人的 raw joint index，而是左右末端各自的
`Δxyz + Δaxis-angle + Δgripper`。状态固定为
`dual_ee_object_relative_state_27d_v2`，显式包含 object→goal、左右 EE→object、对象位移、夹爪、
对象姿态、事件与四个谓词。held-out body 只作为外层 fold 名称出现在审计元数据中，
不拥有 embedding、clock row、action stem 或其他可学习参数。checkpoint 的训练、action
normalization、train-only baseline 和选步只读取另外四个本体。held-out canonical group 的 NPZ
直到这一阶段结束仍保持未打开。

这一步完成后只能说明“共享事件头完成了严格 source-only 训练”。它不会自动授权“跨本体提高
任务成功率”的结论。主结论仍必须来自冻结 actor 的在线配对实验：同一
`(heldout_body, condition, seed)`、同一初始状态、同一有序四候选集合，baseline 执行 candidate
0，ETSF 只在相同四候选中重排；最后由 paired success evaluator 报告二值成功率、阶段进度、
McNemar 和配对置信区间。

## 为什么不能直接拿官方 expert zip 开训

官方 clean/randomized zip 是公开演示数据，但 expert 轨迹本身主要提供正样本，不能独立形成
success/failure/recovery critic 的完整监督。训练入口要求先产生 canonical transition groups，
且 source train 中必须同时存在真实成功和失败标签。失败监督应来自冻结 actor 的 source-body
rollout/candidate branches；不得把 held-out actor 的结果回流到训练或 checkpoint 选择。

因此当前下载完成并不等于训练数据完成。必须依次具备：

1. 11 个官方 `move_can_pot` payload 的 size + SHA-256 全量核验 receipt；
2. 每个本体一个冻结 native actor authority，绑定 checkpoint 和采样合同；
3. 每个本体一个经过无标签解析适配器生成的 canonical manifest；
4. source-body candidate branch 的成功/失败、事件、时长、对象变化监督；
5. 五折各自独立的新输出目录。

缺少任一项时入口 fail-closed，不会退化为“只用 clean expert 正样本”或把 held-out 数据用于补洞。
actor authority 中的 checkpoint 不能只是一个声明哈希：单文件 checkpoint 按文件 SHA-256 校验，
SmolVLA/Hugging Face 目录 checkpoint 则按有序相对路径、字节数和逐文件 SHA-256 计算 tree hash；
目录中的 symlink 或特殊文件会失败关闭。training binding 还必须显式记录本次用户授权只覆盖公开数据、禁止受保护内部数据并且只
允许远端 CUDA 训练。`train-fold` 会检查 CUDA 可用且设备名包含 `4090`，因此不能在本机 CPU
静默启动真实训练。

## 效果导向的训练目标

旧训练只优化逐行六头损失，且每个本体 action stem 独立；目标本体 action stem 没有训练样本时仍是
随机网络。v2 取消这种关节索引别名，并把同一 root 的四个 candidate 始终放在同一 batch：

- event/success/recovery/duration 保留 unweighted proper loss，`sigmoid(success_logit)` 仍可解释为
  成功概率，不使用会改变先验的 class-weighted BCE；
- 增加独立 `candidate_rank_logit`，用同根 pairwise logistic loss 直接学习
  `success > event stage > goal-distance reduction` 的词典序；
- 对象效果使用 Student-t(3) 稳健 NLL，避免旧 Gaussian scale head 的少数异常值把 validation
  object loss 推到上千并拖坏 shared trunk；
- checkpoint 主键改为 source validation 的 body×condition 宏平均 best-of-4 `ΔSR`，其次为宏平均
  selected SR 与 success-changing pair accuracy；六头复合预测分数只作 tie-break；
- `dt` 与 duration 都使用物理秒，duration 不再主导候选重排 checkpoint。

这样训练目标直接对应最终的“candidate 0 与 best-of-4 哪个成功”，而不是靠 AUC/MAE 猜测重排是否
可能有效。

## Canonical group 契约

每个 group 是一个完整的四候选决策，使用 `np.savez(..., allow_pickle=False)` 的纯数组文件，
必须且只能包含：

- `state [N,27]`；
- `actions [N,H,14]` 与 `action_mask [N,H]`；
- current/post/next event id 和 mask；
- duration、observed/censor mask；
- success、recovery 和 mask；
- `object_delta [N,6]` 与 mask。
- `candidate_index=[0,1,2,3]`；
- `dt [N]`，表示 critic 打分前已知的 planned 首段时长，正式合同固定为 `5/15` 秒；真实
  simulator elapsed seconds 仅用于 duration 标签，不能把执行后信息泄漏进 critic 输入。

这里 `N` 必须严格等于 4，四行的 state 与 current event 必须相同。否则无法形成同根排序监督，
训练入口直接拒绝，而不是退化为随机逐行 batch。
source validation 按 `(body, condition, requested_seed)` 切分；同一 reset seed 的全部 query 必须留在
同一 lane，不能把 query 0/5/10/15 拆到 train/validation 后虚高 checkpoint-selection `ΔSR`。

候选 rank 显式拼接 `transitioned`、`clock_hidden` 和 current-event 的 duration
`log_mean/log_scale`，所以 planned `dt=5/15` 对最终 score 有数值计算路径。rank 梯度继续更新
state/action/transition backbone；clock 与 duration 特征在 rank 分支处 detach，只由 proper duration
likelihood 更新，rank loss 不直接改写 clock/duration heads。

每个 body manifest 还必须绑定解析式 adapter：

```json
{
  "kind": "analytic_label_free_canonical_v1",
  "trainable": false,
  "labels_or_outcomes_used_to_fit": false,
  "heldout_supervision_allowed": false,
  "state_dim": 27,
  "action_dim": 14,
  "state_schema": "dual_ee_object_relative_state_27d_v2",
  "action_schema": "dual_ee_se3_gripper_delta_14d_v2",
  "elapsed_time_unit": "seconds",
  "duration_unit": "seconds",
  "event_names": ["e0", "e12", "e3", "e4", "eK"],
  "implementation_sha256": "..."
}
```

这个 adapter 可以解析不同关节顺序、末端执行器和控制频率，但只能使用机器人描述、控制 API、
单位和运动学等无标签信息。若需要从数据拟合尺度，它就不再属于本协议，尤其不能在 held-out
body 上拟合。

每个 manifest、training binding、preflight、checkpoint 和 summary 还必须统一绑定 analytic event
spec SHA `4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee` 及同一个事件实现 SHA；
goal rule 固定为初始 can 相对初始 pot 的 x 符号与 `±0.18m` 偏移，required objects 恰为
`can/pot`。训练标签中的五事件和 state27 relative-goal 与在线 runner 都调用这一实现。旧
`8b1f…` Stage2 规范由成功轨迹拟合，永久禁止进入本五本体 LOBO。

这里仍由旧 SHA `75fc9c6e…` 验证的是既有公开 source-slice materialization receipt，本次修复没有
重新物化官方 slice。它不等于正式 paired 指标预注册；后者使用 corrected v2 SHA
`a4e59f647c520609313e1c9aca03dbb3f770504e0383c66bb619dca94b4c6827`。

## 运行方式

先在远端 4090 服务器做纯元数据和文件完整性预检：

```bash
python scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py \
  --mode preflight \
  --binding /PUBLIC/PREPARED/five_body_binding.json \
  --binding-sha256 SHA256 \
  --held-out-body piper
```

预检通过后训练单折五成员共享头：

```bash
python scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py \
  --mode train-fold \
  --binding /PUBLIC/PREPARED/five_body_binding.json \
  --binding-sha256 SHA256 \
  --held-out-body piper \
  --output /NEW/OUTPUT/lobo_piper \
  --device cuda
```

五个 held-out body 必须各用独立 create-new 输出根。训练产物中的
`task_success_evaluation_authorized=false` 是刻意的：后续还需 label-blind live forward、同序候选
commitment 和完整 paired simulator result，不能用 critic AUC/Brier 替代任务成功率。

## 完整 8000 分支离线消融

`run_robotwin2_five_body_lobo_offline_ablation_v1.py` 只接受完整的
`2000 decision × 4 candidate = 8000` 分支 binding，并固定运行四个 variant：

- `success_only`：只训练 proper success BCE，直接按 success logit 选候选；
- `no_time_duration`：关闭 duration loss，并把 rank 的 clock/duration 输入置零；
- `no_object_effect`：关闭 robust object-effect loss 和 rank target 的 geometric progress；
- `full`：正式完整共享头。

四者使用完全相同的 requested-seed-disjoint split、五折、五成员 seed、每成员 3000 step、eval
间隔、batch 和学习率，不提供小样本或缩减预算开关。每折仍只以另外四个 source body 的
validation 选 checkpoint。入口先完成全部 `4 variant × 5 fold = 20` 次 source-only 训练与选模，
然后才打开 held-out body payload 做只读 posthoc 评估；heldout 指标既不回流 checkpoint，也不选
variant。输出同时保留 source-validation 五成员均值和真正 heldout-body 五折结果。heldout 的
best-of-4 `ΔSR`、selected/oracle SR、pairwise accuracy 按正式推理方式先平均五成员 rank score 再
选候选；success/event/duration/object 预测指标为五成员逐模型指标的算术均值，明确不是
ensemble-calibrated 指标。最终 equal-fold macro 跨五个 heldout body 等权计算。
禁用头的预测指标只作 descriptive 输出，不进入该 variant 的 checkpoint tie-break；例如
`success_only` 的 diagnostic tie-break 只允许 success Brier，不能借 event/duration/object 标签选模。

```bash
python3 scripts/run_robotwin2_five_body_lobo_offline_ablation_v1.py \
  --binding /ABS/full8000_training_binding.json \
  --binding-sha256 FULL_BINDING_SHA256 \
  --output /ABS/new/full8000_lobo_ablation
```

远端 `watch_robotwin2_five_body_postformal_ablation_v1.py` 只负责顺序调度：它必须先看到正式
`1000 pair / 2000 rollout` 报告完整落盘并释放 RTX 4090，才会调用上述完整消融入口。该 watcher
没有小样本、少折、少成员或缩短 step 的参数；它不改评分、不选 variant，也不把消融结果混入
已经冻结的正式 paired 实验。

## 完整 8000 分支到五折训练的远程 watcher

`watch_robotwin2_five_body_branches_to_lobo_training_v1.py` 是正式的断 SSH 后处理入口。它在
CPU 上等待五个 manifest 达到精确的 `5 body × 2 condition × 50 seed × 4 query = 2000`
decision，并逐个核验四候选 `candidate_index=[0,1,2,3]`、NPZ member/shape/dtype、正 planned `dt`、payload
SHA 和无额外文件。watcher 不解释 success/event 等结果数组；每折真正打开的 payload 仍只来自
另外四个 source body。等待期间不持有 GPU，完整采集且 4090 空闲后才顺序启动五折，每折固定
5 个 ensemble member、每 member 3000 step。

四个 query 固定为 `0/5/10/15`。若某个预定 seed 在 query 前已经 terminal，collector 可从
`[2026081000, 2026090000)` 的开发区间补一个新 seed；watcher 验证的是每个
`body×condition×query` 恰好 50 个唯一开发 seed，而不是错误要求五个本体都保留同一段连续 seed。
正式 paired evaluation 的 `2026090000..2026090099` 不在这个区间内。

本次远程正式链路采用：

```bash
nohup /usr/bin/python3 \
  /home/user/etsf_robotwin2_branches_to_lobo_watcher_code_20260830_v2_analytic/watch_robotwin2_five_body_branches_to_lobo_training_v1.py \
  --branches-root /home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic \
  --actor-checkpoint /home/user/etsf_smolvla_models/smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/checkpoints/020000/pretrained_model \
  --materialization-receipt /home/user/public_benchmark_receipts/robotwin2_move_can_pot_5emb_materialization_a967b852_20260830_v1.json \
  --actor-authority /home/user/etsf_robotwin2_fivebody_ee16_actor_authority_full8000_20260830_v2_analytic.json \
  --binding /home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic.binding.json \
  --output-root /home/user/etsf_robotwin2_fivebody_lobo_shared_head_full8000_20260830_v2_analytic \
  --state /home/user/etsf_robotwin2_fivebody_lobo_full8000_20260830_v2_analytic.watcher_state.json \
  --run-exit /home/user/etsf_robotwin2_fivebody_lobo_full8000_20260830_v2_analytic.run.exit \
  > /home/user/etsf_robotwin2_fivebody_lobo_full8000_20260830_v2_analytic.watcher.log 2>&1 < /dev/null &
```

最终 `five_fold_training_summary.json` 对每折保存 source-validation 的 member 值及均值：宏
best-of-4 `ΔSR`、宏 selected SR、候选 pairwise accuracy、success Brier/AUROC、post/next event
F1/accuracy、时长 MAE/NLL、对象 RMSE/NLL 和各自 support。这里的宏 `ΔSR` 是训练与选模证据，
不是 held-out 在线成功率；五本体正式 paired rollout 仍是下一阶段。

## 当前与旧入口的关系

旧 `train_multibody_leave_one_body_out.py` 仍保留历史 Piper/UR5 诊断和
`source_body_clock` 消融。新五本体入口不调用该消融，因为为未知本体预留 clock/body row 虽可置
零，却仍不是最严格的“零目标参数”。新入口只训练 body-agnostic single-row shared head；不同本体
的差异在模型外由受审计的无标签解析 adapter 处理。
