# 共享头暂停点记录（2026-08-30 10:21 CST）

## 1. 已冻结并推送的代码

- 仓库：`/home/jj/Event-Time-Value-Transport`
- 分支：`main`
- 本地 HEAD：`7d4bf86b265830106e6950d25e46b4ab00fd7a48`
- `origin/main`：`7d4bf86b265830106e6950d25e46b4ab00fd7a48`
- 远端共享头 watcher 使用只读代码目录：
  `/home/user/etsf_robotwin2_fivebody_terminal_v10_code_7d4bf86`
- 当前正式采集仍使用已经冻结的 v8 collector 目录：
  `/home/user/etsf_robotwin2_fivebody_terminal_v8_code_93e76da`
- 不要在正式采集未结束时替换 collector、修改正式 root 或中断三个采集进程。

## 2. 远程 4090 正式任务

- 主机：`user@100.115.128.14`
- GPU UUID：`GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`
- 记录时 GPU：100% utilization，17143/24564 MiB。
- 正式数据根：
  `/home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip`
- 进度：70/2000 decisions，280/8000 branches。
- 分本体：
  - `arx-x5`：24（clean 20，randomized 4）
  - `piper`：24（clean 20，randomized 4）
  - `ur5`：22（clean 20，randomized 2）
  - `aloha-agilex`：0
  - `franka`：0
- LOBO watcher：`waiting_for_complete_public_branches`
- 配对成功率 watcher：`waiting_for_true_complete_five_fold_lobo`
- 消融 watcher：`waiting_for_formal_paired_completion`
- 采集和 watcher 都运行在远端，本机关闭不影响它们。

## 3. 当前可以诚实报告的效果

- 共享头尚未开始完整五折训练，因此目前没有可报告的 held-out 预测结果或跨本体 `Δ成功率`。
- 在 63 个 decision 的稳定快照上：59/63（93.7%）存在真实终局事件或终局目标进度差异，说明四候选不是退化的，dense 排序监督有效。
- 同一快照中成功分支为 0，终局只覆盖 e0/e12，没有 e3/e4/eK；主要瓶颈是临界成功监督缺失，不是候选无差异，也不是需要继续增加 gate。
- oracle 平均阶段进度相对 candidate-0 增加约 0.0198（阶段分数的绝对值），但这只是候选 headroom，不是训练后 critic 效果，更不是闭环成功率。
- 最终效果仍必须用 held-out 本体、同 seed 配对的
  `SR(actor+shared-head)-SR(actor)` 与阶段进度差验证。

## 4. 暂停时的未完成改进代码

两个 agent 已被显式中断，工作树保留在可继续审查的状态，没有提交、推送或部署：

- 新文件：
  `scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
  - 目标是用 public RoboTwin scripted expert 只推进到首次 e3/e4；
  - 根以后完全改用同一 frozen actor 的四候选与 actor continuation；
  - 五个预注册 seed 标签盲绑定 H=`10/25/50/100/200`；
  - 计划规模为五本体、两条件、e3/e4、每组合五根，共 100 decisions / 400 branches；
  - 输出独立 supplement root，绝不追加到正式 8000-branch root。
- 修改中：
  `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
  - 增加 source-train-only supplement proper-world stream；
  - 固定权重 0.25；
  - 可训练 next-event/duration/recovery/object/terminal proper losses；
  - supplement 不进入 normalization、baseline、rank/utility、source validation、checkpoint selection 或 calibration；
  - held-out supplement manifest 和 payload 设计为 zero-open；
  - supplement complete groups 正在加入 member-specific Poisson bootstrap，避免 ensemble 不确定性虚低。
- 修改中测试：
  `tests/test_train_robotwin2_five_body_lobo_shared_event_head_v1.py`

这些代码仍处于中间状态，恢复后必须先完成以下接口收口：

1. raw collector manifest 与 trainer supplement manifest/binding 的字段、格式和 SHA 合同尚需完全对齐；
2. 需要独立五本体 supplement binding/materializer，不能手工拼 JSON；
3. collector 的 expert-root snapshot 必须明确结束 planner 语义并证明 fresh actor restore 一致；
4. 需要完成关键测试和静态审计，尚未运行最终回归；
5. 未部署远端，不能启动 supplement 采集；
6. 正式 C-only 流应保留为基线，C+supplement 是独立增强实验，不能覆盖基线结果。

## 5. 工作树边界

以下是用户原有未提交内容，继续时必须保留并排除在本轮提交之外：

- `scripts/collect_openvla_etsf_rollouts.py`
- `scripts/train_openvla_etsf_shadow.py`
- `artifacts/`
- `scripts/launch_smolvla_causal_observer_source63_autonomous_r17.py`

本轮 agent 新增/修改且尚未提交的目标文件只有：

- `scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- `tests/test_train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- 本暂停记录。

## 6. 恢复后的顺序

1. 读取本记录并核对远程采集仍健康，不中断正式 collector。
2. 审完新 expert-root collector，完成 raw manifest → supplement binding 的可执行桥。
3. 审完 trainer 的 held-out zero-open、source-only 数据流和 supplement bootstrap。
4. 只运行与这两个改动直接相关的完整回归，然后运行现有五本体相关套件。
5. 单独提交并 push，排除用户原有 dirty 文件。
6. 部署新的只读代码目录到远端，但等 4090 空闲后再采 100/400 supplement。
7. 完整运行 C-only LOBO/paired 基线，再运行 C+supplement LOBO/paired；比较 held-out 五本体宏平均 `ΔSR`、阶段进度和 95% CI。

## 7. 2026-08-31：scripted-root reserve roster 收口

固定五个任意 reset seed 无法保证 public expert 在五本体、clean/randomized 下都产生 e3/e4。补充 collector 已改成不依赖 outcome 的有序 reserve 协议：

- 每个 `(body, condition, H slot)` 冻结 16 个 seed；五本体合计使用互不重叠的 `2026081000..2026081799`，与 primary `2026082000+` 分离；
- H 仍固定为 `10/25/50/100/200`，manifest 明确区分 `horizon_slot`、H 和真实 `requested_seed`；
- 只选择第一个同时具有 e3/e4 且两个 root 都可 fresh-restore canonicalize 的 seed，选择发生在任何 actor candidate outcome 前；
- 所有 rejected seed 都记录原因，reserve 耗尽非零退出，missing attempt 不再被标为 complete；
- selected root pair 在 outcome 前保存 create-once 恢复 bundle；已有 group 必须同时通过 payload 和 diagnostics SHA 才可跳过；
- 完成规模仍为每本体 20 decisions/80 branches、总计 100/400；materializer 和 upgrade watcher 都按 selected-slot 设计 fail closed，materializer 仍不打开 NPZ。

2026-08-31 后续合并已经把 reserve 协议接入 v9 trainer，接口现状如下：

1. trainer 内部独立实现与 collector 相同的 reserve 常量和 `reserve_roster(body)` 公式，生产训练不 import collector；
2. root-selection、horizon、reserve-roster contracts 已逐项同步；
3. `validate_supplement_body_manifest` payload-blind 校验本体局部 160-seed roster、seed→H map、10 个 selected slots、严格有序 rejection ledger 和 selection-before-outcomes；
4. group identity 已改为 `(condition, horizon_slot, requested_seed, root_event)`；
5. outer-heldout manifest/payload 保持 zero-open，四个 source body 的 roster/selection SHA 从实际 manifest 重算并与 binding 比对。

联合 CPU 回归为 122 passed；只有把该版本提交、推送并部署为新的远程只读代码目录后，supplement 自动流水线才可切换到它。
