# ETSF 标量进度基线审计与入口

## 当前覆盖边界

现有 `train_openvla_etsf_action_q.py` 是终局 critic：输入初始 OpenVLA hidden 和候选首动作块，使用候选分支的终局 success 做 BCE 与组内 pairwise loss。它能够回答“哪个候选最终成功”，但没有标量 progress 监督，也不预测动作后的 future latent。因此，action-success-Q 不能替代下面两个对照：

- `direct`：`state + candidate action -> scalar progress`；
- `latent_future`：`state + candidate action -> predicted future latent -> scalar progress`。

新的 `train_openvla_etsf_progress_baseline.py` 实现这两个轻量对照。它没有 VLAC 的异构视频预训练、生成式 actor/critic 和 RL 微调，也没有 ProgressVLA 的 latent-action expert、diffusion guidance 或 action decoder，所以实验中只能称为 “VLAC/ProgressVLA 风格的同数据轻量消融”，不能称为完整复现。

## 严格对照契约

- 只接收显式指定的 `--train-data` 与 `--validation-data`；CLI 不提供 test 参数，也不会打开 sealed test。
- 可选 `--split-manifest` 只核对 `train` 和 `validation` logical key，不解析 test 样本。
- 支持 schema v4 和 v5。v4 每个候选产生一个首动作块样本；v5 额外使用每条分支的连续 query 链训练晚期 progress 与 future latent。
- v5 的候选 success 排序只比较每个候选的 `query_index=0`；后续 deterministic continuation query 只提供 dense regression 监督，不会被误当作并列候选。
- progress 标签由同一份对象轨迹上的动态可逆谓词压成 `[0,1]` 相位：`e0/e12/e3/e4/eK -> 0/.25/.5/.75/1`。其中仅 successful terminal query 使用终局 success 确认 `eK=1`，所以准确契约是 `terminal_eK_progress_target_only`，而不是“完全没有 success 监督”；success 不会进入非终止 query。如果对象离开目标区域，相位可以回退。
- 模型选点只看 validation progress MAE，success 只作为下游候选排序诊断；损失中没有独立的 success BCE 或 pairwise success loss。因此它与 action-success-Q 的监督边界仍然不同。所有候选排序指标（包括 candidate success AUC）只使用各候选的 `query_index=0`，continuation query 仅用于回归监督。
- 两种基线使用相同的固定随机 hidden 投影和小型动作 GRU，不重复加载或训练 VLA 大模型。

## 训练入口

正式 schema-v5 `train100` 是单一 collection root，不能把同一路径同时传给旧的
`--train-data/--validation-data`。正式入口改用安全 launcher：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/launch_openvla_etsf_progress_v5.py \
  --dry-run
```

dry-run 会等待并验证 structured factual 三成员和正式 counterfactual ensemble 已完成，
复算 `train_openvla_etsf_counterfactual.make_group_splits` 的固定 seed `20260827`
划分，并要求与 counterfactual 的 `split_manifest.json`、ensemble contract 完全一致。
100 组固定划为 train 70 / validation 15 / internal sealed 15。

launcher 在输出的 `split_views/` 下生成 manifest-only view：train/validation manifest
只包含对应 logical key、schema、resolved seed、当前 SHA256 和绝对 HDF5 路径，不复制
HDF5；split manifest 对 internal sealed 15 只记录 logical key，不记录其路径、文件 SHA，
也不会把它传给 progress trainer。正式运行按顺序训练 `direct` 三个 seeds，再训练
`latent_future` 三个 seeds，共六个独立成员；suite summary 报告 validation 三 seed
均值/标准差。actor、selected、oracle 和 success-ranking 诊断只出现在 validation，
train 只报告 progress/future-latent 回归指标，internal sealed 15 完全不评估。

正式 launcher 仅接受 4090/CUDA；在创建不可恢复的输出前以及每个顺序成员之间，
都会通过 `nvidia-smi` 等待 GPU 0 上除 launcher 自身外的 compute PID 清空。dry-run
不探测 GPU、不创建 view/output、不启动训练。完整一致的六成员输出可安全跳过；任何
部分、冲突或失败输出都 fail-closed。完成态会重验实际 split manifest SHA、六条冻结
命令的 SHA、train/validation view manifest SHA、event-spec、优化超参数、validation-only
best step，以及 checkpoint/summary 的 config/contract 镜像；当前 trainer 不支持 resume，
必须使用新目录。

下面的单成员命令仍保留用于合成测试或已经物理拆分好的开发集，不应用于单一正式
train100 root。

直接标量版本：

```bash
python scripts/train_openvla_etsf_progress_baseline.py \
  --train-data /path/to/schema_v5_train \
  --validation-data /path/to/schema_v5_validation \
  --event-spec /path/to/event_spec.json \
  --split-manifest /path/to/frozen_split_manifest.json \
  --variant direct \
  --output /path/to/progress_direct \
  --device cuda
```

future-latent 版本只需改为：

```bash
python scripts/train_openvla_etsf_progress_baseline.py \
  --train-data /path/to/schema_v5_train \
  --validation-data /path/to/schema_v5_validation \
  --event-spec /path/to/event_spec.json \
  --split-manifest /path/to/frozen_split_manifest.json \
  --variant latent_future \
  --output /path/to/progress_latent_future \
  --device cuda
```

正式训练应在指定 4090 上显式使用 `--device cuda`。默认设备是 CPU，防止误在本机启动 GPU 作业。

输出同时报告 progress MAE/RMSE、future-latent cosine（仅 latent 版本）；validation 另报告 candidate success AUC、组内 pair accuracy、NDCG、候选 0/选择后/oracle 成功率及 paired bootstrap CI。后几项仍是离线分支诊断，不能单独证明闭环成功率提高。

## CPU 合成测试

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_openvla_etsf_progress_baseline.py
```

launcher 合成测试：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_launch_openvla_etsf_progress_v5.py
```

测试覆盖 schema v4/v5 loader、v5 continuation query、可逆 progress 标签、两种前向/损失、候选评估、两步 CPU 训练、绝对路径/SHA view、70/15/15 同源划分、六成员 dry-run、prerequisite 等待和冲突输出拒绝。
