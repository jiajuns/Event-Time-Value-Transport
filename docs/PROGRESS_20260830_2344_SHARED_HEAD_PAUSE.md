# 共享头暂停点记录（2026-08-30 23:44 CST）

## 1. 当前结论

- 正式工作只在远程 `user@100.115.128.14` 的 4090 上运行；本机没有跑正式采集或训练。
- 远程现在仍处于 **正式候选分支数据采集阶段**，共享头的完整五折 LOBO 训练尚未开始。
- 暂停时不终止远程 collector 和三个 watcher；关闭本机不会影响远程任务。
- 当前不能诚实声称共享头已提高跨本体成功率。最终主指标仍是 held-out 本体、同 seed 配对的
  `SR(actor + shared head) - SR(actor)`，并同时报告阶段进度、95% CI 与 McNemar 检验。

## 2. 远程 4090 断点

- GPU UUID：`GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`
- 记录时 GPU：72% utilization，14389/24564 MiB。
- frozen actor：
  `/home/user/etsf_smolvla_models/smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/checkpoints/020000/pretrained_model`
- 正式数据根：
  `/home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v4_root_pose_roundtrip`
- 正式 collector 使用只读 v8 代码：
  `/home/user/etsf_robotwin2_fivebody_terminal_v8_code_93e76da`
- watcher 使用只读 v10 代码：
  `/home/user/etsf_robotwin2_fivebody_terminal_v10_code_7d4bf86`
- 总进度：566/2000 decisions，2264/8000 branches。
- 分本体进度（decision / branch）：
  - `aloha-agilex`：92 / 368
  - `arx-x5`：127 / 508
  - `franka`：100 / 400
  - `piper`：128 / 512
  - `ur5`：119 / 476
- LOBO watcher：`waiting_for_complete_public_branches`
- 配对成功率 watcher：`waiting_for_true_complete_five_fold_lobo`
- 消融 watcher：`waiting_for_formal_paired_completion`
- 暂停时 collector 和 watcher 进程均存活。不要替换正式 collector、修改正式数据根或中断这些进程。

## 3. 当前数据证据与真正瓶颈

最近一次完成的 553-decision 审计结果：

- 2212 个 N=4 one-deviation branches。
- 242/553（43.8%）decision 有稠密的阶段/目标进度监督。
- 只有 4 个成功 branch，且全部来自同一个 `arx-x5` decision；mixed-success decision 为 0。
- candidate-0 成功比例与 oracle 成功比例都为 1/553（约 0.18%）。这不是整条 episode 的策略成功率，只是 one-deviation decision branch 统计。
- 平均 oracle 阶段增益为 +0.00904，平均 oracle 目标距离增益为 +0.04165 m；20 个 decision 有阶段正增益，190 个有目标正增益。

因此 N=4 数据能够训练“哪一个动作带来更好事件进度”，但目前没有观测到二值成功可选择空间。下一步有效改进不是继续堆 gate，而是：

1. 加入 expert 仅推进到 e3/e4 根、之后完全由 frozen actor 产生候选和 continuation 的临界成功补充数据；
2. 保留 C-only 正式基线，另训 C+supplement，严格做五本体 LOBO；
3. 用独立 N=8 候选池闭环评估，验证更大的可选择空间能否转化为 held-out `Delta SR`。

## 4. 本地已完成但尚未提交/部署的改进

### e3/e4 临界根补充数据

- `scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
  - public scripted expert 只推进到第一个 e3/e4 根；
  - 根以后完全切换为同一 frozen actor 的候选与 continuation；
  - 五本体、两条件、e3/e4、每组合五根，计划 100 decisions / 400 branches；
  - 输出独立 supplement root，不污染正式 8000-branch root。
- `scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
  - 验证完整 5x2x2x5 设计、actor/checkpoint authority 和 seed 不重叠；
  - 生成不可变 binding；held-out payload 保持 zero-open。
- `docs/ROBOTWIN2_SCRIPTED_EXPERT_ROOT_SUPPLEMENT_V1.md`

### 共享头 supplement 训练路径

- `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
  - supplement 只进入四个 source bodies 的 proper world-model losses；
  - 固定权重 `lambda=0.25`；
  - 不进入 normalization、baseline、rank/utility、source validation、checkpoint selection 或 calibration；
  - held-out supplement manifest/payload zero-open；
  - member-specific ordinary logical-group Poisson bootstrap，避免 ensemble 不确定性虚低。

### 独立 N=8/N=16 候选池评估

- `scripts/run_robotwin2_five_body_postformal_candidate_pool_v1.py`
  - 默认 N=8；可选 N=16；不修改现有正式 N=4 基线协议；
  - baseline 固定执行 raw candidate-0，共享头才做风险调整重排序；
  - 五成员分数为 `mean - 0.5 * population_std`；
  - 完整计划 1000 个配对 / 2000 个 rollout，输出 Delta SR、阶段进度、bootstrap CI 和 McNemar。

## 5. 验证状态

- 2026-08-30 23:43 CST 已运行 collector、trainer、postformal candidate-pool、正式 collector 和正式 paired runner 的联合测试。
- 结果：`90 passed, 1 warning in 1.82s`。
- 唯一 warning 是本机 PyTorch 尝试初始化 CUDA 时发现本机驱动版本较旧；该批测试成功，正式 GPU 工作仍只在远程 4090 环境执行。
- 新代码尚未完成最后人工审计、提交、push 或远程部署；恢复时不能把它误当作已上线任务。

## 6. Git 与工作树边界

- 仓库：`/home/jj/Event-Time-Value-Transport`
- 分支：`main`
- 当前本地 HEAD 与 `origin/main`：`7d4bf86b265830106e6950d25e46b4ab00fd7a48`

用户原有 dirty 文件，恢复时必须保留并排除在本轮提交之外：

- `scripts/collect_openvla_etsf_rollouts.py`
- `scripts/train_openvla_etsf_shadow.py`
- `artifacts/`
- `scripts/launch_smolvla_causal_observer_source63_autonomous_r17.py`

本轮待审查/提交文件：

- `scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- `scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
- `scripts/train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- `scripts/run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- `tests/test_collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- `tests/test_train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- `tests/test_run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- `docs/ROBOTWIN2_SCRIPTED_EXPERT_ROOT_SUPPLEMENT_V1.md`
- 本暂停记录。

注意：两个新测试文件受仓库 `.gitignore` 的 `tests/` 规则影响，提交时需要显式 `git add -f`。

## 7. 恢复顺序

1. 先读取本记录并只读核对远程 4090 collector/watcher 是否健康。
2. 等正式 2000 decisions / 8000 branches 完整结束，让现有 C-only LOBO、paired success 和 ablation 流按 watcher 顺序运行。
3. 审核 supplement collector/materializer/trainer 的 manifest、SHA、held-out zero-open 和 seed-overlap 合同。
4. 审核 N=8 runner 的 paired reset、candidate commitment、动态候选轴和结果收据。
5. 跑五本体 watcher/ablation/paired 的更广回归；只提交本轮文件并排除用户 dirty 文件。
6. push 后创建新的远程只读代码目录；不要覆盖 v8/v10。
7. 4090 空闲后再采 100/400 supplement，跑 C+supplement 五折 LOBO，并用同 seed N=4/N=8 paired rollout 报告真正跨本体效果。
