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

- 每个 optimizer step 使用两个互不替代的数据流：uniform complete-decision batch 只训练
  event/success/recovery/duration 的 unweighted proper loss 与 object Student-t(3) NLL；另一个
  balanced rank-only batch 只训练 group listwise utility。`sigmoid(success_logit)` 因此仍可解释为
  成功概率，不使用会改变先验的 class-weighted BCE，也不让 mixed oversampling 改写概率先验；
- v5 的 `candidate_rank_logit` 不再读取自由的 `transitioned` latent 或 `clock_hidden`。排序头只接收
  28 维显式 predicted-consequence 向量：post-event 五类概率、next-event 五类概率、成功概率、恢复
  概率、当前事件 duration 的 `log1p` 分布 mean/scale、moving-object SE(3) effect 的六维
  mean/scale，以及由 `state[0:3]` relative goal 与预测 object translation 计算的目标距离进展和
  Student-t(3) 径向不确定性。目标进展严格定义为
  `||relative_goal|| - ||relative_goal - predicted_object_translation||`；不确定性用 object
  translation scale 的 delta-method 径向标准差；
- 上述 consequence feature 在进入 utility MLP 前整体 stop-gradient。listwise loss 只学习如何组合
  已校准的事件、结果、时间和对象预测，不会把 proper prediction head 或 shared transition 重新训练成
  绕开事件语义的隐式 critic；这些预测头仍由各自 proper likelihood 和 robust object-effect loss
  训练；
- 同根四候选只要同时存在成功/失败，就直接最小化“softmax
  分给任一成功候选的概率质量”的负对数；不再让大量 failure/failure pair 淹没真正改变成功率的
  监督；
- 全失败 decision 才以 `0.1` 权重学习完整 continuation：先严格限制在最大的
  `terminal_max_event_id` 层，层外目标质量恒为零；层内按冻结的 `0.02m` goal-progress softmax
  温度分配 listwise 目标（`no_object_effect` 在最高层均匀）。不再把标签乘 `100/10` 相加，也不把
  `1e-6m` 数值抖动硬标成唯一最优；
- 对象效果使用 Student-t(3) 稳健 NLL，避免旧 Gaussian scale head 的少数异常值把 validation
  object loss 推到上千并拖坏 shared trunk；
- 五个成员不再各自挑一个 raw-logit 最优 checkpoint。每个成员保存相同 eval step 的 source-only
  快照，然后用正式部署同一个 decision 内标准化五成员 ensemble 联合选择一个共同 step；主键为
  body×condition 宏平均 best-of-4 `ΔSR`，其次依次为 mixed-success decision 的选中成功率和
  success-changing pair accuracy；只有这些同分时才看全失败 dense continuation 排序，五成员六头
  复合预测分数均值最后破同分；
- `dt`、duration 与 `event_age_seconds` 都使用计数 simulator step 得到的物理秒。事件年龄是“当前规范
  事件自最近一次进入以来的已持续时间”，在候选执行前可得，同根四候选完全相同；它和 planned `dt`
  一起进入 proper duration 分布，再由 duration mean/scale 间接进入 utility，不存在
  `clock_hidden → rank` 的自由直通；duration 也不再主导候选重排 checkpoint。

这样训练目标直接对应最终的“candidate 0 与 best-of-4 哪个成功”，而不是靠 AUC/MAE 猜测重排是否
可能有效。

五成员的 uniform proper 数据流仍使用原始 decision-group Poisson bootstrap。rank-only 数据流对每个
mixed-success decision 使用 `1 + Poisson(1)`，保证五个成员都看到每个稀有成功改变比较，同时保留
成员间权重差异；全失败 decision 继续使用可为零的 `Poisson(1)`。rank sampler 在
`body × condition × current_event` 间轮转，尽量让每个 batch 同时包含 mixed 与 dense group，且同一
logical group 绝不在同一 batch 重复。该重采样只作用于 utility listwise loss，不读取 held-out，
也不改变 proper success/event 概率的训练分布。

## Canonical group 契约

每个 group 是一个完整的四候选决策，使用 `np.savez(..., allow_pickle=False)` 的纯数组文件，
必须且只能包含：

- `state [N,27]`；
- `actions [N,H,14]` 与 `action_mask [N,H]`；
- current/post/next event id 和 mask；
- duration、observed/censor mask；
- success、recovery 和 mask；
- moving object 的 SE(3) `object_delta [N,6]`（`Δxyz + shortest Δaxis-angle`）与 mask；
- 完整 continuation 的 `terminal_max_event_id / terminal_stage_progress /
  terminal_goal_distance / terminal_goal_progress [N]`；前两者满足
  `terminal_stage_progress=1 if success else terminal_max_event_id/4`，后两者和 root state 的
  goal residual 一致；
- `candidate_index=[0,1,2,3]`；
- `dt [N]`，表示 critic 打分前已知的 planned 首段时长，正式合同固定为 `5/15` 秒；真实
  simulator elapsed seconds 仅用于 duration 标签，不能把执行后信息泄漏进 critic 输入。

这里 `N` 必须严格等于 4，四行的 state 与 current event 必须相同。否则无法形成同根排序监督，
训练入口直接拒绝，而不是退化为随机逐行 batch。
source validation 按 `(body, condition, requested_seed)` 切分；同一 reset seed 的全部 query 必须留在
同一 lane，不能把 query 0/10/20/30 拆到 train/validation 后虚高 checkpoint-selection `ΔSR`。

候选 rank 只拼接显式 consequence prediction。planned `dt=5/15` 与当前物理事件年龄先经过隔离
clock 形成 proper current-event duration 分布，再以该分布的 mean/scale 间接进入 utility；`transitioned` 和
`clock_hidden` 本身都不进入排序特征。整块 28 维 feature 在 rank 分支处 detach，因此 rank loss
只更新 utility MLP，不直接更新 state/action/transition backbone 或任何 event、success、recovery、
duration、object predictor；这些模块只由其 proper/robust 监督更新。

五成员部署聚合也不再直接平均不可比的 raw logit。对每个 decision、每个 member 的四个分数先减去
该 member 的候选均值，再除以四候选 population std；std 不大于 `1e-6` 的常数成员贡献全零，最后
始终对恰好五个成员等权平均。该纯函数与合同由 trainer 导出，source validation、离线 heldout 和
正式 runner 必须共用，禁止跨 decision 计算均值/方差或让某个 logit 尺度大的成员支配选择。
训练 summary/checkpoint 同时绑定 trainer 文件 SHA、共同选择 step、全部 source-only 候选 step 的选择键，
正式 runner 会拒绝五成员 step 不一致或并非由当前训练实现产生的 checkpoint。

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
- `no_time_duration`：关闭 duration loss，并把 consequence utility 的 duration mean/scale 两维置零；
- `no_object_effect`：关闭 robust object-effect loss 和 rank target 的 geometric progress，并把 utility
  的 object mean/scale、predicted goal progress/uncertainty 共十四维置零；
- `full`：正式完整共享头。

四者使用完全相同的 requested-seed-disjoint split、五折、五成员 seed、每成员 3000 step、eval
间隔、batch 和学习率，不提供小样本或缩减预算开关。每折仍只以另外四个 source body 的
validation 选 checkpoint。入口先完成全部 `4 variant × 5 fold = 20` 次 source-only 训练与选模，
然后才打开 held-out body payload 做只读 posthoc 评估；heldout 指标既不回流 checkpoint，也不选
variant。输出同时保留 source-validation 五成员均值和真正 heldout-body 五折结果。heldout 的
best-of-4 `ΔSR`、selected/oracle SR、pairwise accuracy 使用与正式部署相同的“五成员各自在同一
四候选内 z-score 标准化后等权平均”合同，避免 raw rank logit 尺度不同造成单成员支配。
success、post/next event、duration、object-effect、recovery 均先把五个冻结成员在概率或密度空间
组成部署 ensemble，再计算 Brier/NLL/ECE/AUROC、macro-F1、秒制 MAE、Student-t(3) mixture NLL
和 recovery Brier/AP 及逐阈值 precision--recall 曲线；没有 recovery 正例时必须显式标记 PR
不可用，不能填 0。上述预测在打分前还会拒绝任何非有限输出，不再用“五个成员各算一次指标再
平均”冒充 ensemble 质量。
duration NLL 同时报告 observed density、censored survival 和两者合并后的总 mixture NLL；五折宏
平均还必须附每项实际有支持的 fold 数，缺标签的 fold 不得暗中当作零。

不确定性只来自同一五成员的预测分歧。报告按 success/event/duration/object/recovery 给出
error--risk-coverage 与 AURC，并按候选决策给出 rank 与 success argmax/pairwise 分歧、成员 rank
选择分歧下的 selected-failure/oracle-regret risk-coverage。预测 observation unit 是 candidate
branch，排序 unit 是完整四候选 decision，依赖 cluster 明确为
`heldout_body×condition×requested_seed` 下的全部 query/candidate；最终 equal-fold macro 跨五个
heldout body 等权计算。该 posthoc 消融不构造置信区间，也不得据其 heldout 结果选择 variant 后再
把同一 heldout 当作确认性证据。
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

四个 query 固定为 `0/10/20/30`，覆盖 `max_steps=200` 闭环的早/中/后段，而不是只采前
15 个 policy query。若某个预定 seed 在 query 前已经 terminal，collector 可从
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
