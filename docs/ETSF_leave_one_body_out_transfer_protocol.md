# ETSF 严格 leave-one-body-out 跨本体迁移协议

入口：`scripts/train_multibody_leave_one_body_out.py`。

这个协议回答的不是“多个本体联合训练后，各自在 validation 上是否准确”，而是更严格的
问题：**完全不读取某个目标本体的训练样本与标签，仅用其他本体训练和选模，最终在预先
冻结的目标本体 development lane 上是否仍能预测事件、时间、成功和对象变化。** 当前允许
分别留出 `piper` 和 `ur5-wsg`；两者必须独立训练、独立报告，不能把其中一个目标 dev 的
结果用于另一个实验的选择。

## 数据边界

程序先只根据 `(body, policy, task, seed)` 做与标签无关的确定性分层划分，再生成五条互斥
lane：

- `source_train`：除目标本体外的训练组；唯一允许拟合权重、动作归一化和统计 baseline 的
  lane；
- `source_validation`：除目标本体外的 validation 组；唯一允许选择 checkpoint 的 lane；
- `target_development`：目标本体原本已分到 validation 的组；仅在两个模型变体的全部十个
  checkpoint 都选定后打开一次并终评；
- `target_unused_train`：目标本体原本分到 train 的组；保持未打开，绝不改作 adaptation；
- `sealed_test`：所有本体的 test 组；HDF5 永不打开，也不报告 test transition 数。

split 必须先以独立 JSON 冻结，并在训练时同时提供文件 SHA-256。程序会重新扫描 label-free
identity，与冻结文件逐字段比较；文件字节、内部逻辑 SHA 或当前输入绑定任一不同都会失败。
所有活动输入、split 和输出路径在解析符号链接后如包含 `fresh` 或 `confirmation` 都会被拒绝。

## 模型与基线

每个实验顺序训练两个各含五成员 Poisson group-bootstrap ensemble 的变体：

1. `source_body_clock`：source body 的 duration clock 系数可以学习；目标本体只占一个预留
   clock row，该 row 初始化为零且梯度强制为零，因此没有目标适配；
2. `body_agnostic`：所有 source/target body 共用同一个 clock row，是明确的 no-body-
   conditioning ablation。

其余 canonical semantic、每动作 schema stem 和多任务 heads 与
`train_multibody_canonical_event_world_model.py` 相同。若某个动作 schema 在 source train 中
完全不存在（留出 Piper 时，当前数据的 OpenVLA schema 即属于这种情况），该 schema 的
归一化固定为 mean=0/std=1，并在 receipt 中记为
`unseen_source_schema_frozen_identity`；不会用 Piper dev 动作拟合。相应 stem 也没有得到
source supervision，因此这一项应解释为诚实的 zero-shot policy/schema 缺口，而不能宣称已
证明 action-conditioned 跨策略迁移。

此外报告只由 `source_train` 拟合的统计 baseline：majority post/next event、per-event/global
duration median、empirical success 和 zero object delta。目标本体不存在的 body-duration key
不会回退到任何目标统计，只能使用 source per-event/global 值。

## 输出指标

在冻结的 target development 上输出 `global`，并按 body、policy、task 分解：

- post event / next event accuracy、macro-F1、10-bin ECE、ensemble mutual information；
- observed duration MAE、ensemble 标准差及其与绝对误差的相关性；
- success AUROC、Brier、NLL、10-bin ECE、ensemble 标准差和正负样本 support；
- 6D object delta RMSE、ensemble 标准差及其与逐行 RMSE 的相关性。

object RMSE 使用每条 target row 中由对象位姿构造出的几何 delta，即便目标动作向量不可用也
能评估；输出显式标记 `geometry_delta_all_target_rows`。这不意味着无动作的 UR5 row 曾监督
object-effect head。AUROC 遇到单类目标时输出 `null` 和
`unavailable_single_class`，不会伪造 0.5。

结果同时包含 `source_body_clock` 相对 source-only statistical baseline、
`body_agnostic` 相对 baseline，以及前者相对 no-body ablation 的逐指标差值。正向指标
（F1/AUROC）差值越大越好，误差指标（MAE/Brier/RMSE）差值越小越好。

## 可审计运行顺序

本机只做无数据 CPU smoke：

```bash
python3 scripts/train_multibody_leave_one_body_out.py --mode synthetic-smoke
```

正式实验应在 4090 上分别运行 Piper 和 UR5。以下用 Piper 示意；所有输入 SHA 必须替换为
本次冻结资产的实际值。

先 preflight（不打开任何 rollout/group payload）：

```bash
$PY scripts/train_multibody_leave_one_body_out.py \
  --mode preflight \
  --held-out-body piper \
  --stage1-root "$STAGE1_ROOT" \
  --stage1-source-manifest "$SOURCE_MANIFEST" \
  --stage1-source-manifest-sha256 "$SOURCE_SHA" \
  --stage1-target-manifest "$TARGET_MANIFEST" \
  --stage1-target-manifest-sha256 "$TARGET_SHA" \
  --event-spec "$EVENT_SPEC" \
  --event-spec-sha256 "$EVENT_SPEC_SHA" \
  --openvla-schema5-manifest "$SCHEMA5_MANIFEST" \
  --openvla-schema5-manifest-sha256 "$SCHEMA5_SHA"
```

用同一组参数把 split 写入一个全新路径：

```bash
$PY scripts/train_multibody_leave_one_body_out.py \
  --mode freeze-split \
  --held-out-body piper \
  ...相同输入绑定参数... \
  --split-plan-output /new/immutable/piper_lobo_split.json
sha256sum /new/immutable/piper_lobo_split.json
```

最后训练。`--split-plan-sha256` 必须是上一步文件的字节 SHA；输出目录必须不存在：

```bash
$PY scripts/train_multibody_leave_one_body_out.py \
  --mode train \
  --held-out-body piper \
  ...相同输入绑定参数... \
  --split-plan /new/immutable/piper_lobo_split.json \
  --split-plan-sha256 "$FROZEN_SPLIT_FILE_SHA" \
  --output /new/immutable/piper_lobo_train \
  --device cuda
```

UR5 将三处 `piper`/Piper 输出路径改为 `ur5-wsg`/UR5 新路径。不能复用训练输出目录，也不能
修改已冻结 split。最终主要文件为：

- `protocol_receipt_before_payload_open.json`：任何 payload 打开前写出的协议收据；
- `<variant>/source_selection_summary.json`：只含 source-validation 选模信息，明确记录 target
  尚未打开；
- `lobo_training_summary.json`：全部 checkpoint 选完后才生成的 target-development 终评。

本脚本不部署、不中途访问 sealed test，也不把该预测指标自动解释为机器人任务成功率提升。
任务成功率仍需后续在同一冻结候选集上做 paired baseline-vs-plugin 执行评测。
