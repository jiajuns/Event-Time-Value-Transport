# Fresh-50 一次性确认流水线

入口：`scripts/launch_openvla_etsf_fresh50_confirmation.py`。它只接受 reset-only
预注册并解析完成的 50 个 fresh seeds、冻结的 counterfactual ensemble 和固定 event
spec，严格按 collection → one-shot evaluation → progress baseline 的顺序串行执行。

正式默认依赖：

- fresh manifest：`/home/user/etsf_event_world_model_code_20260827/artifacts/protocol/fresh_confirmation_seeds_20260827.json`
- counterfactual ensemble：`/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry1_20260827`
- fresh 输出：`/home/user/etsf_openvla_fresh50_confirmation_move_can_pot_20260827`

先运行同一命令并追加 `--dry-run`。dry-run 完成全部 CPU 契约审计、打印三个冻结
argv，不创建输出、不探测 GPU，也不读取 fresh outcome 标签。

## 消耗 fresh 前的硬门

launcher 在启动 collector 前验证：恰好 50 个 ordered requested/resolved seeds，候选池
与官方 registry SHA，reset-only audit 顺序，无 official/development split 重叠，以及
ensemble aggregate/member/checkpoint SHA 镜像。冻结 validation 还必须同时满足：

- `guard.enabled=true`；
- `scoring_selection.selected_candidate_id` 唯一对应的候选具有
  `passes_pre_guard_evidence_gate=true`。

任一条件失败都会在 collection 和 one-shot reservation 之前拒绝。当前若 validation
得到 disabled guard，不能用 fresh 数据诊断、改阈值或再次选模；应保留 fresh 未消费状态。

也可显式传入 five-fold OOF authorized final 目录。该模式不接受普通 OOF 中间产物：
必须验证 final `training_summary.json` 对 ensemble manifest 的 SHA、selection 文件 SHA
及内部 canonical 签名、五折 summary/raw artifact SHA、OOF authorization 与冻结
guard/scoring/calibration 镜像；且只能进入带 fresh manifest 的 one-shot evaluator。
原 validation-split ensemble 契约仍兼容，但 disabled guard（包括 retry1）继续拒绝。

## 标签隔离与恢复

collector 使用 `--seeds-file ... --seeds-key test --allow-unregistered-seeds
--fresh-seed-manifest ...`，并额外原子写出 label-free `collection_identity.json`。
launcher 在 evaluator 预约前只读该 identity、HDF5 attrs 和 `candidate_names`；不读
`manifest.json`、success/event/trajectory datasets，也不计算 HDF5 内容 SHA。

collection 中断可用完全相同契约和显式 `--resume` 安全恢复。evaluator 通过
`evaluated_once.json` 原子预约；预约后中断禁止重跑。只有完整 confirmatory marker
校验通过后才调用 progress launcher。完整流水线再次调用只在 evaluation、progress
summary 和 progress audit 的 SHA/状态全部一致时安全跳过；其他部分输出一律拒绝。

正式运行还要求 CUDA 设备名包含 `4090`，各 GPU 阶段串行等待 GPU 0 空闲。Python
argv 保留 venv 软链接路径，不解析到缺少 venv 包的 base interpreter。
