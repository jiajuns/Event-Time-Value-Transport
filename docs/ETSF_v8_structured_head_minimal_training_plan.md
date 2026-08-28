# ETSF 下一版结构化预测头最小训练与严格验收方案

状态：基于冻结 development250/v6 OOF 的只读开发审计，面向 v7 之后的独立
structured-head repair。本文不修改 v7、action-rank guard 或 Fresh50 合同，也不把已反复
查看的 development250 重新解释为确认集。

## 1. 结论

现在不应再次联合微调 shared world-model core。v6 说明事件表征并未整体失效，失败集中在
各自的目标、尺度和概率训练合同：主 `next_event` 已有明确 heldout skill，应 bit-exact
冻结；duration 应先使用已经显示稳定收益的 event×body median residual shrinkage；success、
predicate、regress/recovery 应改成互不干扰的 detached-feature 概率 adapter；object learned
delta 在获得可信的 pose-quality 标签前只保留 robust/zero fallback，任何新 object adapter
都只能是 fail-closed 的探索头。

这套最小方案可以用现有监督先训练 duration、success/outcome、predicate 和 exploratory
recovery adapter；但 object 的物理准确性、稀有 destination/stationary/recovery 的可靠估计、
跨任务/跨 policy/跨本体结论都需要新采数据。即使在现有 250 groups 上重新做 OOF，结果也
仍是 development evidence；旧 100 groups 还存在 frozen factual head 历史训练重叠未排除，
严格的下一次确认必须使用有 base-training exclusion provenance 的新 groups。

## 2. 冻结 v6 的证据

所有数值来自五折 owner-heldout raw prediction；没有读取 Fresh50。

| 域 | v6 结果 | 当前判断 |
|---|---|---|
| dynamic `next_event` | accuracy `0.9209`，NLL `0.2469`，macro-F1 `0.5068`；相对 persistence 和 other-fold prior 的 NLL、top-1 error clustered 95% CI 均严格小于 0 | 主头通过，禁止为修别的头而重训 |
| observed next-reached event | accuracy `0.8883`，NLL `0.3773`，macro-F1 `0.4923`；class support `[0,1906,222,2,180]`，class 3 F1 为 0 | 稀有目的地不可作逐类准确主张 |
| post predicates | lifted Brier skill 通过；moved/near_goal/stationary/success 未通过 paired Brier；正例分别为 `7636/54/27/133`（lifted 为 `4882`） | 排序信号存在，但多数概率头不合格 |
| duration | observed `2310`、right-censored `6372`；原头 log1p-MAE `0.6649`，event×body median 为 `0.6271`，paired CI `[-0.0073,0.0915]` | 原头失败 |
| duration repair | 每个目标折都只在其余四折选到 residual multiplier `0.375`；heldout log1p-MAE `0.5531` vs baseline `0.6271`，paired CI `[-0.0902,-0.0518]` | 当前唯一有清晰 heldout 收益的 repair；仍仅是 development 信号 |
| object delta | MAE/coord `0.06783` vs zero `0.05996`，paired CI `[0.00685,0.00803]`；4204/8682 行精确为零，9 行 max-abs delta > 5 | learned object 失败 |
| object repair | q99.5 fold-local robust mask 后五折都选择 multiplier `0`；退回 baseline 后没有 learned skill | 下一版默认 fallback，不授权 learned output |
| binary success/outcome | outcome NLL `0.5099` vs prior `0.3641`，paired CI `[0.0569,0.2179]`，虽 macro-F1 较高仍失败；first-four success 严格概率校准因 factual training weight 未记录而不可评 | 需要新的 unweighted probability adapter |
| within-group success ranking | 62 个可比较 groups，equal-group pair accuracy `0.5403`，error-vs-random CI `[-0.1304,0.0524]` | 未显著胜随机 |
| recovery | terminal regress `106`、recovery `57`；当前 factual config 为 `recovery_supervised=false`；开发 adapter AP `0.154` vs prevalence `0.0496`，但 Brier/NLL skill CI 均跨 0 | 有排序信号，没有概率 skill |
| uncertainty | next-event/duration/object/success 的 AURC 均优于 random 且五折方向一致 | 可作为 risk ordering；不能替代概率校准或预测准确性 |

一个重要的 outcome 语义问题是：57 个 recovery 中有 55 个同时最终 success。把
`failure/success/recovery` 当互斥三分类会用 recovery 覆盖 success 标签；下一版应改为
`p(success)` 与 `p(recovery | regress)` 的多标签/条件分解，failure 直接取
`1-p(success)`，而不是继续训练互斥三分类。

## 3. 现有监督与必须新采的标签

| 目标 | development250 已有监督 | 能否立即训练 | 要形成可靠主张还缺什么 |
|---|---|---|---|
| next-event | 8682 条 structured transition，类支持 `[1034,7486,17,12,133]` | 可以，但主头已经通过，最小方案是不训练 | 若要声称每类可靠，需要定向增加 e3/e4；若要跨任务，需要新任务的 event mapping 和轨迹 |
| next-reached destination | 2310 条 observed duration 对应的 destination，类支持 `[0,1906,222,2,180]` | 只能训练常见类 | class 3 等稀有目的地必须补采；没有支持的 class 0 不能作准确率主张 |
| reach / duration | 2310 observed + 6372 right-censored，均有 horizon 内是否到达标签 | 可以训练 `p(reach within horizon)` 和 observed-only conditional duration | 若要估计 horizon 之外的完整 survival curve，必须采更长 continuation 或记录可比较的 censor horizon |
| predicates | moved/lifted/near_goal/stationary/success 全部从完整 pose trajectory 派生 | moved/lifted/success 可立即训；near_goal 可探索；stationary 支持不足 | stationary 每折正例只有 `6/5/5/4/7`，需预注册地补采而不是看指标停采 |
| terminal success/failure | first-four deployment candidates 共 1000 条，success 116；全部 terminal 共 1150 条，success 133 | 可以训练新的 unweighted binary probability head | 第五候选只可进训练和 appendix；新的独立 groups 才能确认校准/控制收益 |
| regress/recovery | terminal regress 106、recovery 57；标签由完整动态 phase 轨迹按 persistence contract 派生 | 可以训 exploratory conditional adapter | recovery 每折为 `12/3/20/9/13`，不足以五折稳定评估；至少补到每折 conditional recovery/non-recovery 各 10 |
| object delta | 8682 条 pre/post object pose delta | 只能用 robust heuristic 做探索 | 必须新增 `object_pose_valid`、active-object identity、单位/坐标系和最好有 contact/grasp validity；仅用 q99.5 不能判断大位移是真实还是 simulator corruption |
| 跨 policy/body/task | 当前 manifest 只有 `move_can_pot|piper_piper_0.6` | 不能 | 同 task 同 body 换 policy、同 task 同 policy 换 body，以及多任务完整轨迹；每条轴必须独立采集，不能把 policy 与 body 同时变化称为跨本体证据 |

这里“必须新采”有两种含义：recovery/stationary/rare event 已有标签定义，只缺按定义产生的
新正负样本；object 还缺一个真正的新质量标签。前者不能用 oversampling 创造信息，后者也
不能用分位数 mask 冒充人工或 simulator provenance。

## 4. 最小可训练结构

### 4.1 冻结边界

1. semantic encoder、transition core、现有 next-event 主头、action-rank residual 和 guard
   全部冻结；保存训练前后 SHA，要求 bit-exact。
2. 每个 repair head 只接收 `transition.detach()`、冻结 logits、event/body id 和 fold-local
   baseline，不允许一个坏头通过 shared gradient 破坏已通过的 next-event。
3. 每个 binary probability head使用 unweighted BCE。若 batch sampler 为稀有类过采样，loss
   必须乘 inverse-sampling weight 恢复原始分布；不得再用未记录的 `pos_weight` 后把 sigmoid
   直接解释为概率。
4. bias/temperature 只能在目标 outer-training split 内的真正 inner-OOF prediction 上拟合；
   其他 outer OOF prediction 不能用于目标折校准。

### 4.2 各头最小改动

- **next-event**：直接透传 frozen logits。若新任务/本体必须适配，只允许 zero-init 的小型
  additive residual adapter，并要求其对 frozen head 同时满足 rare-class 改善和 overall
  non-inferiority；当前任务不为追求更高点估计重训该头。
- **duration**：预测形式固定为
  `training-fold event×body median_log1p + 0.375 * frozen_residual`。`0.375` 是下一独立数据
  上预注册的常数，不在 target fold 或新确认集上再选。若进一步训练 adapter，其 target 是
  observed-only log1p residual，loss 为 Laplace NLL；6372 个 censored row 只训练独立 reach
  BCE，不再把 conditional duration location 向 horizon 外推。
- **predicates**：五个独立 linear/one-hidden-layer detached adapter，分别用 unweighted BCE；
  不共享最后一层，不让 moved 的大支持压过 stationary。stationary 在支持门满足前输出
  `not_evaluable`。
- **success/outcome**：只训练一个 terminal `p(success)` adapter，failure 为补概率。训练可以
  使用注册的第五候选，但部署主指标和阈值拟合只看前四候选；第五候选结果单独写 appendix。
  排序若需要单独 residual head，不能把 pairwise loss 混入概率 head。
- **regress/recovery**：先训练 `p(regress)`，再只在真实 regress rows 上训练
  `p(recovery | regress)`，unconditional recovery 为二者乘积。两头都从 frozen feature
  `detach()`；当前 57 例只允许 development exploratory，不授权部署。
- **object**：部署点预测保持 outer-training robust median（当前数据等同 zero fallback）。
  可并行训练一个两段式探索 adapter：`p(nonzero valid delta)` + quality-valid nonzero row 上的
  Student-t(df=3) residual；只有严格胜过 zero 和 robust median 两个 baseline 时才激活 learned
  output，否则 multiplier 固定回 0。没有显式 pose-quality 标签时不得写“准确物理变化”。
- **uncertainty**：分类用预测熵与小 adapter ensemble disagreement，回归用条件 scale 与
  ensemble disagreement；uncertainty 只用于 abstention/risk ordering，不能用于事后删除坏样本
  来抬高主准确率。

## 5. 严格 OOF 和 provenance

1. outer fold 仍按 logical group 固定五折。每个 target fold 的 baseline、normalization、
   object mask、class sampler、early stop、threshold、calibration 都只能来自 outer-training
   groups；需要模型选择时，在该 outer-training split 内再做 inner group split。
2. 每折保存 head owner fold、outer-training logical-key SHA、base checkpoint SHA、base 的训练
   group provenance、实际 loss 权重、label-derivation SHA、normalization 和 mask contract。
   缺任一项就将概率校准和 strict adequacy 标为 unavailable。
3. frozen factual head 对 legacy old100 的历史标签重叠未被排除，因此全 250 的新 adapter OOF
   只能作开发分析。严格结果只可来自证明未被 base 训练使用的 groups，或者重新构造真正
   outer-excluded 的 base；不能只靠 adapter 自己没见 holdout 来消除 feature leakage。
4. 同一份 development250 已用于根因分析、shrinkage 和门槛设计。下一次在它上面的结果仍
   不能称 prospective confirmation；门槛必须在新独立 development expansion 标签开放前签名。
5. success/outcome/ranking 的主表只包含每组前四 deployment candidates。training-only 第五
   候选可以参与 outer-training，但不能提高 deployment 支持数、准确率、pair accuracy 或
   adequacy decision。

## 6. 下一独立评估的固定门槛

所有 loss 差异以 logical group 为 cluster、10000 次 bootstrap、双侧 95% CI。要求
`model - baseline` 的 CI 上界严格小于 0；只看点估计不算通过。任一必需项不可评即该域
fail-closed。下面门槛是下一独立数据的 prospective proposal，不能倒签为 v6 门槛。

| 域 | evaluability | 必须同时通过的 skill / calibration 门 |
|---|---|---|
| next-event | 每个被主张的类总支持至少 10 且五折均出现；逐类 publication claim 要总支持至少 25、每折至少 3 | NLL 与 top-1 error 分别严格胜 persistence、other-fold smoothed prior；macro-F1 严格胜两 baseline；multiclass Brier skill CI 上界 < 0；ECE-10 ≤ 0.10；逐类 F1 全报 |
| observed destination | 同 next-event；当前 support=2 的 class 3 自动 not-evaluable | NLL/top-1 error/macro-F1 胜 observed-only other-fold prior；ECE-10 ≤ 0.10 |
| reach | 每折正负各至少 10 | Brier、NLL 分别胜 other-fold Beta(1,1) prevalence 且 paired CI 上界 < 0；PR-AUC-prevalence 的 group-bootstrap 下界 > 0；ECE-10 ≤ 0.10 |
| conditional duration | 至少 30 个有 observed label 的 logical groups；被主张的 event×body cell 在 outer train 至少 20 observed | observed log1p-MAE 胜 crossfit event×body median；Laplace NLL 胜 training-fold median+MAD Laplace；至少 4/5 folds 点估计不劣；censored rows 只进入 reach 指标，不伪造 duration error |
| 每个 predicate | 全局正负各至少 50，且每折正负各至少 5 | Brier、NLL paired skill 均通过；AP-prevalence bootstrap 下界 > 0；macro-F1 胜 prevalence classifier；ECE-10 ≤ 0.10；否则该 predicate 单独 not-evaluable 并使完整 predicate-domain fail |
| success/failure | 前四候选中每折正负各至少 10 | Brier、NLL paired skill 通过；AP-prevalence bootstrap 下界 > 0；macro-F1 胜 prevalence；ECE-10 ≤ 0.10；若用于选动作，within-group pair error-vs-random 的 CI 上界 < 0 |
| regress | 每折正负各至少 10 | 与 success 相同的 Brier/NLL/AP/macro-F1/ECE 门 |
| recovery given regress | 每折 recovery 与 non-recovery 各至少 10；当前数据不满足 | conditional 和 unconditional recovery 都须 Brier/NLL 胜 crossfit prevalence；AP-prevalence bootstrap 下界 > 0；macro-F1 胜 prevalence；ECE-10 ≤ 0.10 |
| object delta | 至少 30 groups；显式 quality-valid coverage ≥ 99%，invalid 单独报告 | valid 与 all-recorded 两个口径下 MAE 均严格胜 zero 和 crossfit robust median；RMSE 不劣两个 baseline；95th-percentile absolute error 不劣；至少 4/5 folds 点估计不劣；任一失败 multiplier 回 0 |
| uncertainty | 每折对应任务都有 error 与正确样本 | AURC-improvement-over-random 的 group-bootstrap 下界 > 0，且五折 AURC 均优于 random；Spearman(error, uncertainty) > 0；最高 uncertainty quartile error > 最低 quartile |

## 7. 最小执行顺序

1. 先签名上述数据范围、base provenance、adapter 结构和门槛；不修改 v7。
2. 用现有监督做一次 adapter-only 五折 development OOF：next-event passthrough、duration
   fixed shrink、predicate/success/regress/recovery detached adapters；object 只训练探索分支且
   默认 fallback。无需 full-core GPU 微调。
3. 结果不通过时不调 guard、不反向改门槛。按预注册配额补采 stationary、rare destination、
   regress/recovery 和 pose-quality 数据；收集停止条件只看 support，不看模型指标。
4. 在与 frozen base 训练集严格不重叠的新 groups 上做一次独立确认。先过各预测域门，再讨论
   可插拔 selector；只有闭环同 seed success delta 的 95% CI 下界不小于 0，才可声称提高任务
   成功率。
5. 跨 policy 与跨 body 分开建数据轴；在每条轴的预测门和闭环门都通过前，只能写“接口支持
   adapter”，不能写“已跨本体准确预测”。

## 8. 已实现的 schema5→v8 物化合同

实现入口为 `scripts/materialize_openvla_etsf_v8_oof_inputs.py`。它分两步执行：

1. `--preregister-owner-manifest` 只用 label-free identity/HDF5 root attrs 固定 SHA256 五折，
   并绑定 complete `manifest.json`、`collection_identity.json`、每个 HDF5 digest、event spec、
   factual checkpoint、v7 seed manifest/preregistration 及其已签名 exclusion registry 摘要。
   它不会重新打开 Fresh50 exclusion 文件。v8 方案形成于 v7 D250 采集启动后，因此产物固定
   标为 `adaptive_development_only`、`prospective_claim_for_v8=false`；真正 prospective 的 v7
   fixed-policy 评估与它分开。
2. 正式 materialize 只有在 collection/identity 都为 `complete` 时才运行。它先完成 owner
   split，再按折只加载 outer-training labels、拟合 duration/object fallback，最后才把 holdout
   写入独立 evaluation artifact。当前仍为 `collecting` 的服务器集合会 fail-closed，不能提前
   物化。

每个 signed record 保留 frozen transition、duration、next-event/next-reached logits、对应标签、
真实 `aleatoric_uncertainty`、物理尺度 object delta 和全部 masks。单 checkpoint forward 没有
epistemic/total uncertainty，因此合同明确标记 unavailable，禁止用 0 或 aleatoric 冒充 ensemble
total。schema5 也没有 object-pose quality 字段，因此 quality 是 unavailable/fail-closed；不能用
`dense_mask` 冒充。duration 额外保存 outer-train observed-only MAD/`log(2)` Laplace scale 合同。

物化目录先在 sibling temporary directory 完整生成，再一次性 rename；payload 内容哈希覆盖
labels、masks、group/role/fold、frozen tensors、baseline 和 fallback。训练 CLI 必须同时提供 signed
materialization manifest 与 outer fold id，并验证完整五折十个 train/holdout artifact，拒绝孤立或
partial `.pt`。standalone 自哈希 base-exclusion JSON 永远不能把状态升级为 `proven`；在 factual
checkpoint 没有权威训练身份合同前仍保持 development-only。

adapter 训练仍是固定 group 顺序的 AdamW 接口，记录顺序、完整 loss trace 和
`convergence_status=not_assessed_fail_closed`；CPU one-step/one-epoch 只证明接口、梯度隔离和 hash
不变，不能称为已收敛或最优。后续严格 OOF 应使用 full-batch convex optimizer 或预注册收敛门。
