# SmolVLA（Aloha-trained）→ Piper + V7 严格配对开发协议

状态：协议冻结器与合成测试已实现；尚未生成 530-seed 目标 manifest，未授权执行、采集、
训练或跨本体结论。本协议只使用非 Fresh 开发数据。

## 1. 这个实验回答什么

目标 cell 是：Aloha 数据训练的 SmolVLA actor 在 Piper 仿真中执行 `move_can_pot`。
在同一组 requested/resolved seeds 上比较：

- baseline：SmolVLA 直接策略，始终执行命名为 `deterministic` 的 candidate；
- plugin：同一 actor 生成四个固定噪声候选，由冻结的 V7 事件世界模型选择器选候选；任一
  contract guard 失败即回退 `deterministic`。

如果主成功率与预测门同时通过，最多可以表述为：**V7-assisted 系统在本次预注册的非 Fresh
开发实验中，改善了 Aloha-trained SmolVLA 在 Piper 上的执行效果**。

这不是“事件世界模型跨本体”的隔离实验。V7 的事件核心本来就来自 OpenVLA+Piper，目标执行
仍是 Piper；这里核心跨的是 policy/state/action 接口，actor 跨的是 Aloha→Piper。要证明世界
模型自身跨本体，必须把 Piper 完全排除出核心训练，保持 policy/task 不变，只允许少量预注册
Body/Clock adapter，再在未触碰的 Piper seeds 上通过相同门槛。

## 2. 当前必须披露的四个运行时事实

远端只读审计确认：

1. Piper 的 `observation.state[14]` 是 controller 的 commanded `drive_target`，不是 measured
   joint qpos。协议分别哈希两者做初态配对，selector 不能把 state14 叙述成实测 proprioception。
2. 一个 policy decision 经 TOPP 展开为可变数量的 250 Hz physics steps。因此当前 duration
   只能称为 **decision-step duration**；没有 simulator monotonic timestamp 和逐 decision
   physics-substep count 时，禁止称物理时间或跨本体时钟迁移。
3. `SubEnv` 显式 seed reset 不会刷新 `episode_info_list`，因此其 instruction 不保证来自当前
   seed。两条件必须使用显式冻结的同一 instruction，并为每个 pair 保存独立语义校验 receipt；
   禁止从陈旧 `episode_info_list` 获取指令。
4. 当前 V7 输入是 OpenVLA 4096D，而 SmolVLA 共享 prefix 是 960D；forward smoke 通过也不能
   让 V7 直接吃 960D。最初 20 个 operational smoke 只跑 direct actor。V7 plugin 只有在
   960D StateAdapter、`50×14` PolicyAdapter、Piper BodyActionAdapter、decision-step ClockAdapter
   和非特权 PredicateAdapter 全部验证后才可执行。

## 3. 可安全绑定的 D250 字段

冻结器只接受 collector 生成的 `collection_identity.json`，绝不接受含 outcome 的
`manifest.json`。可绑定字段包括：

- task/body/schema/candidate count；
- 250 个 requested/resolved seed 及组顺序；
- 四个候选的固定名称；
- event-spec SHA；
- V7 seed-manifest 和 preregistration SHA；
- 每个逻辑 group 的路径和完成状态。

identity 的 `label_access_contract` 必须为
`identity_only_no_success_steps_event_or_outcome_fields`。递归出现 success、reward、event-id、
event steps、duration、recovery、selected-index 或 prediction 字段会失败。D250 只绑定源证据，
不计入目标样本量。

## 4. 530 个目标 seeds 与阶段

目标 reset-only manifest 固定三个互斥 split，requested 和 resolved identity 均不得与 D250 或
其他 split 重叠：

| 阶段 | groups | 允许操作 |
|---|---:|---|
| adaptation | 80 | 前 20 只做 direct actor、标签隔离的运行 smoke；通过后按固定前缀释放 80 条，可训练小 adapter/observer，shared core 不变 |
| validation | 50 | 只能从预注册网格选一个 adapter checkpoint、uncertainty threshold 和 candidate-validity threshold |
| evaluation | 400 paired seeds | 全部在线 selection record 签名后一次性打开标签；禁止中期有效性检验、改模型、改阈值或扩样本 |

400 的规划依据是 net success delta 5pp、总 discordance 12%、双侧 α=.05 时，配对符号检验的
正态近似 power 约 .82。它只是样本量规划，不替代最终 exact test；看到 outcome 后禁止重新估计
样本量。

Fresh identities 不由本冻结器打开。目标 manifest 只接受一个不披露 identity 的交集为零
attestation，绑定 heldout identity-set SHA 与 target identity-set SHA。

## 5. 同 seed 配对和在线无标签选择

每个 evaluation pair 的两个完整 episode 必须具有相同 requested/resolved seed、显式 instruction
hash、initial scene hash、measured-qpos hash 与 commanded-drive-target hash。运行顺序由
`sha256(pair_id || 20261003)` 固定随机，不能手工决定。

direct 与 plugin 的 candidate0 noise/RNG 完全一致。plugin 在每个 query 只能读取当前 RGB 或
actor-hidden observer、commanded drive target、语言、四个尚未执行的 action chunks，以及冻结
世界模型的事件预测/不确定性。禁止读取 success、reward、future event/object pose、candidate
outcome、另一个 condition 的结果和 privileged simulator object pose。

每次 `env.step` 前必须不可变写入：pair/query id、四个 candidate action SHA、observer input SHA、
prediction SHA、contract guards、proposed/selected index 和 fallback reason。evaluation evaluator 在
打开第一个 outcome/event label 前以 `O_EXCL` 预约；所有 400 pair 的 selection logs 未完成时不得
解封。

不允许 outcome-based retry。基础设施失败最多在任何 outcome 反序列化前按同 seed 重试一次；
evaluation broken pairs 超过 2% 则整个实验无效，并同时报告 ITT worst-case missing sensitivity。

## 6. 固定指标与门槛

唯一 primary comparison 是 plugin vs direct 的 400-pair unconditional success delta。

同时满足才算 task-success gate 通过：

- delta 严格大于 0；
- 20,000 次 paired-seed bootstrap 的 95% 下界严格大于 0；
- helpful/harmful discordant pairs 的 exact 双侧 McNemar/sign-test `p < .05`；
- 至少改变 40 个 episode，selection coverage 至少 10%；
- 所有 changed episodes 中 harmful rate 不高于 10%。

事件预测按 logical episode group bootstrap，不能把 transition 当 IID：

- next event：accuracy、macro-F1、NLL、ECE；
- observed destination：accuracy、macro-F1、NLL、ECE；
- decision-step duration：log1p MAE、Laplace NLL、coverage；
- baseline 只在 adaptation 上拟合：event frequency、current-event identity、
  event×body decision-step median。

要声称预测准确，next-event accuracy 和 NLL 对最强 baseline 的 group-bootstrap 95% 下界均须严格
改善，且报告类别必须在 evaluation 有非零支持。不能只报 pooled accuracy 掩盖 rare classes。

当前 recovery/success heads 均关闭且不得进入 ranking。recovery 至少需要 50 个独立 positive groups
和 50 个 negative groups 才允许训练；不足时只报告 support。满足支持门后仍需 Brier/NLL 优于
adaptation prevalence、AP-prevalence、ECE 和 unconditional paired recovery delta，不能靠 recovery
指标挽救失败的 primary success test。

## 7. 防止自适应偏差

- adaptation 固定前缀，不能按 outcome 或 uncertainty 选样；
- validation 明确计为模型选择，不计 test evidence；
- evaluation size/order 固定，无 interim effectiveness stop；
- evaluation outcomes 不得修改候选数、adapter、公式、margin、threshold 或 exclusion；
- secondary metrics 不能把失败的 primary 改写成成功；
- 所有排除、重试和缺失 pair 必须逐项报告。

## 8. 冻结器

实现：`scripts/preregister_smolvla_piper_v7_paired_development.py`。

它绑定 D250 identity、530-seed target manifest、SmolVLA→Piper forward-only preflight、V7
composite activation、event spec、V7 utility、actor-agnostic plugin 和 Smol schema5 collector 的
SHA。输出状态明确为 `execution_not_authorized`；冻结协议本身不会启动 actor、simulator 或训练。

只有另行完成运行时执行器、安全检查、五个 adapter contract 和不可变 selection logger 后，才可
为该冻结协议制作单独的 execution plan。当前不具备该授权。
