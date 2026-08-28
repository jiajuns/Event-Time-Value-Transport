# ETSF 多本体 canonical event world model

入口：`scripts/train_multibody_canonical_event_world_model.py`。

该入口把 Stage1 的跨本体对象几何与 OpenVLA schema-5 动作分支合并到统一的五事件空间：
`e0, e12, e3, e4, eK`。Stage1 的 `e1/e2` 均映射为 `e12`。canonical state 固定为
27D（几何 14D、任务 one-hot 6D、事件 one-hot 5D、可逆 predicate 2D），semantic
state 固定为 96D。

动作并不共享一个含义模糊的 14D stem。Aloha、ARX 与 OpenVLA 各有独立的 temporal
stem，再投影到共享 96D action-effect 空间。Piper/UR5 的 Stage1 rollout 没有动作记录，
其 action effect 精确为零；这些行仅监督事件语义、clock/duration 与 success，不监督
object-effect 或 conditional recovery。

body identity 使用显式 fail-closed alias 表：`piper` 与 schema-5 的
`piper_piper_0.6` 均规范为 `piper`，Aloha、ARX、UR5 保持各自正式名称；任何未登记
拼写直接报错。protocol receipt 会记录本次实际观察到的 raw→canonical 映射、各 raw
标识的 group 数和映射 SHA。

动作在进入各 schema stem 前使用独立 mean/std 归一化。统计只读取 train rows 中
`action_mask=true` 的有效步；缺动作、padding、validation 和 test 均不参与。receipt 与
每个 checkpoint 同时保存每个 schema 的 mean/std、train row/group/valid-step 数和统计
SHA，mean/std 也作为模型 buffer 写入 state dict，恢复 checkpoint 时不会重新拟合。

validation 报告 post/next event accuracy 与 macro-F1、observed duration MAE/NLL、success
Brier/AUROC（单类明确记为 `unavailable_single_class`）、object RMSE/NLL，以及每个头的
有效 support。所有比较基线也只由 train rows 拟合：majority post/next event、按
body+current-event（带 per-event/global fallback）的 duration median、empirical success、
zero-object-delta Gaussian。每 100 steps 在 validation 上计算一次预注册 composite；先最小化
相对 train-only baseline 的五项平均误差，再按 next-event F1、success Brier、duration NLL、
object NLL 和更早 step 依次破同分。每个 ensemble member 只保存该规则选择出的 best
checkpoint，不以最后一步冒充最佳结果；test 不参与拟合、选择或报告。

模型输出：

- post-chunk event 与 next-reached event；
- 含右删失 likelihood 的 log-normal duration；
- success probability；
- 仅在“发生回退且动作可观测”时启用的 recovery probability；
- moving-object xyz 与 relative-goal xyz 的 6D Gaussian delta；
- 五成员 logical-group Poisson bootstrap 的 epistemic disagreement。

## 协议门

所有输入都必须同时传入预期 SHA-256。程序拒绝实际路径或符号链接解析结果中含
`fresh`/`confirmation` 的路径。split 在打开 episode/group HDF5 前，只用
`(body, policy, task, seed)` 构造 logical group，并在每个 body/policy/task stratum 内划分。
训练只把 train/validation descriptor 传给 loader；test HDF5 不打开，test transition 数保持
unknown。

先在本机做无数据 CPU smoke：

```bash
python3 scripts/train_multibody_canonical_event_world_model.py \
  --mode synthetic-smoke
```

正式数据 preflight（只检查绑定、身份和 split，不打开 group HDF5）：

```bash
PY=/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python
$PY scripts/train_multibody_canonical_event_world_model.py \
  --mode preflight \
  --stage1-root /home/user/etsf_stage1 \
  --stage1-source-manifest /home/user/etsf_stage1/source_manifest.json \
  --stage1-source-manifest-sha256 5326b676c43f87dbb02c5b45dc9575aec44dc510fc7a309619b197985bda9e1c \
  --stage1-target-manifest /home/user/etsf_stage1/target_rollouts/target_rollout_manifest.csv \
  --stage1-target-manifest-sha256 89293270eb6590f96732e3ff58292fb81098eeeb22ff4ce7991651df9bcfe794 \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --event-spec-sha256 8b1ff070ee7f9519707f45c42209b266a515b45c95d14f13bb4a95f283f2bff5 \
  --openvla-schema5-manifest /home/user/etsf_openvla_event_branches_v7_development250_20260827/manifest.json \
  --openvla-schema5-manifest-sha256 b611d6604ecd323e90dea15c1bfb6bed6fb22bf974cf77b52e8eb02c0409ef23
```

把 `--mode preflight` 改为 `--mode train`，再增加一个全新的 `--output` 路径即可训练。
默认是五成员 ensemble、每成员 3000 steps、CUDA。该入口不会使用 Stage3 或旧 OpenVLA
checkpoint 做不兼容的 strict load；第一版从 canonical supervision 重新训练，避免 7-event/
5-event、clock64/clock48、state27/state4096 的静默错配。
