# ETSF development100/250 五折 OOF 开发协议

## 目的与证据边界

旧的固定 validation 只有 15 个 intervention groups，其中 deterministic baseline 成功
3 组、四候选 oracle 成功 4 组，只有 1 组存在成功率改善空间。全 train100 则有 baseline
12/100、oracle 32/100、20 个 oracle-headroom groups 和 29 个 mixed-outcome groups。
因此固定 15 组不足以稳定开发候选排序，但这不允许降低 guard，也不允许提前查看 fresh50。

旧实验把 train100 全部定义为 **development data**，做固定、group-level 的 5×20 OOF。
在该轮 fail-closed 后，又通过 reset-only 预注册新增了 150 个与 official150/fresh50 均不
重叠的开发场景。合并后的 development250 固定做 5×50 OOF：每折使用其余 200 场景训练，
三个成员完成后才首次读取该折 50 个终局标签；每个场景恰好产生一次 heldout prediction。
旧100与新150分别保留 4/5 个候选，不做 padding，也不把 fresh50 混入。新增的
`sample_blend_1.000` 只作为组内训练增强；OOF 的 temperature、scoring、guard、headroom 和
成功率授权严格裁剪到 fresh50 同样的
`deterministic/.250/.500/.750` 四候选，防止依靠部署中不存在的动作获得虚假提升。OOF 只提供
development authorization，最终成功率结论仍只来自预注册的一次性 fresh50 confirmation。

## 冻结训练契约

- 三成员 seeds：`20260827/20260828/20260829`。
- development100 每成员固定 100 steps；development250 固定 250 steps，不使用 outer
  holdout early stopping。步数与 group 数同比缩放，使 batch=8 时的每样本期望曝光次数不变。
- 预算在任何 expanded OOF prediction 产生前冻结。依据是 pre-OOF `retry2b` 开发曲线：两个成员
  在 step 100 达到自身最佳组内排序，第三个成员虽在 step 500 达峰，但三成员共同指标从
  step 100 后总体退化；继续使用原 3000-step cap 会明知地外推到过拟合区间。
- pairwise 只比较 success-changing pairs；listwise 只使用二元成功标签。同成功候选不再因
  terminal steps 产生主排序偏好，效率由独立 duration head 监督。
- event、relative transition、predicate、duration、object delta、future latent 监督保留。
- expanded 模型新增 baseline-relative action-effect residual；候选0残差严格为0，绝对场景
  难度仍由事实 success head 建模，组内排序监督只学习动作相对效果。OOF、离线验证与在线
  `predict_candidates()` 使用同一 adjusted success logit 和调整后的 aleatoric uncertainty。

## 授权门

五折的 100 或 250 个 OOF predictions 合并后，才冻结 success temperature、7 项 scoring grid 和
最多 3×3 guard grid。除原 proposal/coverage/LCB/harmful-rate 门以外，还要求：

- 全体 oracle-headroom groups 至少 10；
- guarded groups 至少 10，helpful changes 至少 5；
- unconditional group bootstrap 95% CI 严格大于 0；
- exact one-sided sign/McNemar 通过最多 63 次搜索的 Bonferroni 校正。

任何一项失败都输出 `guard.enabled=false`，停止 final refit 并禁止 fresh50。通过时才在全
全部开发组按同一曝光预算 refit 三成员；fold 模型到 full refit 的温度/不确定性
尺度漂移仍被明确视为待 fresh50 检验的假设。

## 预测精度证据与成功率授权分离

新 OOF fold raw row 额外保存 held-out transition 的完整概率预测，`select` 阶段据此生成独立的
`oof_prediction_diagnostics.json`。它覆盖成功概率 Brier/NLL/ECE、next-event accuracy/NLL、
observed/right-censored duration likelihood、对象 delta 误差、failure/recovery 支持以及
uncertainty-error 关系；详见 `docs/ETSF_oof_prediction_diagnostics.md`。这些指标不写入
`oof_selection.json`，不改变已经冻结的 scoring/guard 搜索或 fresh50 授权门。

旧 train100 OOF raw artifact 只含候选级 event progress，不能据此计算真实 next-event accuracy；
评估器会保留成功概率诊断，并把结构化部分明确标记为 legacy partial evidence。

## 安全执行

入口为 `scripts/launch_openvla_etsf_counterfactual_oof_v5.py`。launcher 只接受全新输出根，
永久保留 output lock；按 `preregister → fold0..4 → select → authorized-only final` 同步串行。
每个 GPU 阶段前必须确认目标 GPU 名含 4090 且没有并发 compute PID。每阶段独立日志、原子
state 和 artifact SHA 均写入输出；部分失败不可 resume，必须换新目录。launcher 不接受
fresh seed/data 参数，也不会自行启动 fresh50。

development250 的服务器入口为
`scripts/launch_openvla_etsf_development250_oof_pipeline.py`。它只等待签名为
`training_ready` 的 250-group 审计，验证 100×4 + 150×5 = 1150 branches、源根和 event-spec
SHA，随后以 hard link 原子合并并冻结每个 HDF5 SHA；确认 RTX4090 空闲后才调用上述 OOF
launcher。审计、合并或 guard 任一步失败都 fail-closed，fresh50 仍禁止。

## 训练前 action-signal baseline

在正式非线性 OOF 前，`diagnose_openvla_etsf_action_signal_oof.py` 对同一固定五折运行了三个
不参与 guard 选择的轻量 baseline。action-delta logistic 为 9%（相对 actor -3pp，
bootstrap CI `[-9pp, 3pp]`）；state×action interaction logistic 为 11%（-1pp，
`[-8pp, 5pp]`）；预注册 pairwise variant 为 12%（0pp，`[-7pp, 7pp]`）。三者的组内
success-pair accuracy 分别为 46.4%、41.2%、43.3%。这说明当前固定线性/PCA 特征没有可靠
动作排序信号，但不能否定 action-conditioned nonlinear world model；同时它预示正式 OOF
很可能 fail-closed，不能把 oracle 32% 误写成可实现成功率。

## 2026-08-27 正式 OOF 结果

远端 RTX 4090 正式输出：

```text
/home/user/etsf_openvla_counterfactual_oof_v5_successonly100_20260827
```

五折全部完成，每折 heldout 的 baseline/oracle 分别为 `2/5、3/9、3/8、3/6、1/4`，合计
仍为 12/32。最好的未加 guard scoring 是 `full_light/full`：84 次非 baseline proposal，
helpful 10、harmful 4，成功率由 12% 到 18%（+6pp），但 proposal paired-delta 的单侧
90% LCB 为 `-0.00115`，仍低于预注册的 0；其余 scoring 的 LCB 更低。没有 guard threshold
通过开发门，最终状态为 `stopped_guard_not_authorized`，因此未执行全100 final refit，
fresh50 继续禁止。该结果是有希望但证据不足，不能报告为成功率已经改善；合理下一步是增加
官方环境中自采的、与 official150/fresh50 均不重叠的 development counterfactual groups。
