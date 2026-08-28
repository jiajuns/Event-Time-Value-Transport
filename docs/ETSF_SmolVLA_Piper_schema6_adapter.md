# SmolVLA→Piper schema6 embodiment adapter

入口：`scripts/train_smolvla_piper_schema6_embodiment_adapter.py`。

该入口只接受签名的 `etsf_smolvla_piper_schema6_training_manifest_v1`，并强制同时提供聚合器
生成的 `etsf_smolvla_piper_schema6_expected_manifest_split_v2` 及其文件 SHA。manifest 中每个 group
必须显式提供 logical group、requested/resolved seed、task、`body=piper`、
`policy=smolvla`、相对路径和文件 SHA。trainer 在任何 HDF 字节哈希或 `h5py.File` 前验证 expected
receipt、trainer 自身 path/file SHA、manifest/target partition/external split 的文件与逻辑 SHA，
以及精确互斥、全覆盖的 60 train / 20 internal-validation / 50 sealed-test identity。它直接使用
external split，持久化 `frozen_group_split.json` 后才打开 train/validation HDF；target
validation50 必须与 sealed test 完全相等，其 HDF 在本入口中永不打开、也不做字节哈希。
formal CLI 已删除 split seed/fraction 参数，不能重新随机分组；`synthetic-smoke` 仍可独立运行。

canonical event spec 是正式训练的强制输入。trainer 不再从 e0/e12/e3/e4/eK 的序号
臆造 moved/lifted/near-goal/stationary/success；它从每条 branch 保存的 object pose 按冻结
calibration 重建逐帧 reversible predicates，并要求重建的 dense event 与 HDF 完全一致。

source checkpoint 必须带完整 `etsf_dual_reserved_rows_source_only_counterfactual_v1`
证明。Piper 使用既有 `__reserved__piper` body row 1；policy 明确使用并冻结 source
`smolvla` row 0，`__reserved__openvla` row 1 只审计、绝不作为目标 policy。
共享 core/head 全冻结。可训练量仅为零初始化的 960D residual low-rank state adapter、
identity 初始化的 14D diagonal/low-rank action adapter、Piper body 单行，以及隔离的
clock beta/log-step-scale。每次 optimizer step 后都会恢复并验证其余 core 和两个 policy 行
bit-exact。

逐 transition 的 event/time/success/object loss 之外，trainer 将每个 logical group 的合法
root candidates 保持成组，绑定 original candidate index、lowest-legal baseline 和 root
intervention 后的最终 branch success。baseline-relative pairwise loss 与 successful-candidate
listwise loss 都先在组内平均、再按 logical group 等权平均；验证使用完全相同的 root score、
最终 outcome 和 lowest-index tie-break 语义。

success 的逐 transition 监督同样是 eventual final branch success，而不是几乎只在终点为真的
`trajectory_success[1:]` 脉冲；trainer 会先验证 trajectory diagnostic 单调且 terminal 与 branch
final outcome 一致，再把同一个最终 outcome 绑定到该 branch 的所有 action-conditioned rows。
这使 dense success loss、源 core 的成功语义和 root candidate ranking 不再互相冲突。

`transition_next_event_id` 在 duration right-censored 行只是 collector 写入的 current-event
占位符，并非观测到的 self-loop。因此 destination/next-event CE、validation accuracy 和每类
support gate 都严格只使用 `duration_observed=true` 的行；post-event 仍使用全部真实一步转移。
`e0` 只在 step 0 定义初态，不是 `dense_event_targets` 的未来 milestone，因此 next-reached
support 只要求结构上可达的 `e12/e3/e4/eK`，同时仍记录 e0 计数用于审计。

恢复概率不是从未受该监督的 source-core recovery 输出迁移而来。trainer 在主 adapter 选模并
冻结后，单独训练一个 `p(recovery | operational regress)` 线性 head：事件低于此前峰值连续
三个保存状态才算 regress；随后连续三个状态回到旧峰值（或到达 eK）才算 recovery；右删失且
尚未恢复的样本不当作负例。该 head 在公开的 semantic transition 表示上强制 stop-gradient，
使用独立 optimizer，且 train 和 internal-validation 必须各自至少有 10 个正类、10 个负类
独立 logical group，否则明确 disabled。未完成独立校准和激活门前，它不进入主 utility 或
uncertainty 聚合。

正式 support gate 默认每个 `body|policy|task` stratum 至少 30 个 train、20 个 validation、
50 个 sealed test group，并同时要求正/负 outcome group、discordant paired group、五类 event、
observed/censored duration、object supervision 和四个 original candidate index 的最低支持。
所有门均为显式 CLI 参数；仅单元/合成自检可把这些参数显式调低，低门结果不得作为正式训练收据。

正式训练在构造 adapter 之前固定 Python、NumPy、Torch CPU/CUDA RNG，并强制 deterministic
algorithms/CuDNN 契约。trainable parameter audit 要求精确八个参数张量集合，而非仅检查前缀。

CPU 自检：

```bash
python3 scripts/train_smolvla_piper_schema6_embodiment_adapter.py --mode synthetic-smoke
```

正式训练（输出目录必须不存在）：

```bash
python3 scripts/train_smolvla_piper_schema6_embodiment_adapter.py \
  --source-checkpoint /ABS/source_member_best.pt \
  --schema6-manifest /ABS/schema6/manifest.json \
  --expected-manifest-split-receipt /ABS/schema6/schema6_expected_manifest_split_v2.json \
  --expected-manifest-split-receipt-file-sha256 64_HEX_SHA256 \
  --canonical-event-spec /ABS/event_spec.json \
  --output /ABS/new_immutable_output \
  --device cuda:0
```

可选 canonical teacher 只需额外给 checkpoint；强制 event spec 同时绑定 manifest 与 teacher：

```bash
  --canonical-teacher-checkpoint /ABS/canonical_member_best.pt
```

启用后，程序从同一 schema6 object pose/registry 重建 canonical 27D，要求重建事件与
HDF dense event 完全一致，再用冻结 teacher 的 96D semantic 作为 alignment target；任一
契约或 SHA 缺失都会 fail closed。

每个成员训练结束会重载实际选中的 `best.pt`，只导出 adaptation80 内部 20-group validation
的 NPZ，绝不打开外部 target-validation50。导出 receipt 明确绑定：duration 为
`log1p_decision_steps`，next-event mask 为 `duration_observed`，success 为 eventual final branch
outcome；object head 的标准化均值/尺度会反变换到物理 `delta_xyz_m`，并记录源 object
normalization 的逻辑 SHA。当前 v1 calibration 为避免把部分无效维度当真值，仅在选中对象的
所有 xyz 都有效时启用该 row，并在 receipt 中显式记录这一 conservative policy。
同一导出还提供 `regress/recovery/recovery_observed` labels 与 `recovery_logit` prediction，
并记录正负 group support、训练/禁用状态、stop-gradient 以及“未进入 utility/uncertainty”契约。
