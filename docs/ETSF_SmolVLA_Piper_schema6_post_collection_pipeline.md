# Schema6 130组之后的可恢复五成员流水线

## 自动化边界

`scripts/launch_smolvla_piper_schema6_post_collection.py` 是断线常驻、create-once、fail-closed 的后采集 watcher。固定顺序为：

1. 等待并认证 Phase2 `adaptation80+validation50` terminal receipt；
2. 仅从签名 preregistration、reset/group receipt、registry/pose JSON 物化 60/20/50 manifest，聚合阶段不打开 HDF 字节；
3. 将五个 source ensemble member 分别训练成五个 adapter；失败 attempt 原样保留，`detach-resume` 创建新 attempt，不覆盖旧证据；
4. trainer 只打开 adaptation60+internal-validation20，target-validation50 和 evaluation400 均不打开；
5. 五个 adapter 冻结后，独立的 `evaluate_smolvla_piper_schema6_target_validation50_ensemble.py` 才被授权一次性打开 target-validation50，生成共同 labels 和五成员 predictions；
6. calibrator 只消费上述 NPZ，冻结事件/持续时间指标、不确定性分解、head support 与 abstain threshold；
7. 只有 `post_event/next_event/duration` 三个核心 head 及 abstain threshold 全部通过，才生成 400 对独立授权请求；watcher 自身永远不执行这 400 对。

evaluation400 的身份、HDF、label、outcome 不进入 manifest、evaluator authority 或 calibration authority。后采集 watcher 只等待一个不披露 sealed identity/outcome 的独立 authority，验证后发布“交给外部 paired launcher”的 handoff receipt，执行计数始终为 0。

## 修复的真实契约缺口

- Phase2 原生产物是 `per_seed_reset_receipt.json + completed_group_receipt.json`，并非旧版 `collection_authority/manifest/final_receipt`；materializer 现可直接验证原生 Phase2 lineage。
- 完成后的 collection 树会被冻结，manifest 不可能写入 HDF 的祖先目录；manifest 现绑定已认证的绝对 HDF 路径，不复制、不软链、不在聚合阶段读取字节。
- trainer 现在重载 best checkpoint 后导出内部诊断 NPZ，并明确绑定 `log1p(duration)`、observed-only next-event、eventual branch success、物理米 object space 及 normalization SHA。
- 正式 calibration 不再错误使用 internal-validation20。该做法无法满足至少 50 validation groups 的 abstention 门，代码中被明确拒绝。
- next-event 指标只在 `duration_observed` 行计算；censored self-loop 不作为标签，结构性初态 `e0` 不进入 future-event support minimum。

## 恢复与GPU互斥

watcher 以新 session detach，等待 PPID=1；训练/evaluator 前持有指定 flock，并要求两次连续 RTX4090 空闲检查。子进程运行时发现非本 stage 后代的 GPU PID 会终止当前 attempt。已签名完成的 member/evaluator/calibration 会在 resume 时重新验 SHA 后跳过；不完整 attempt 不删除、不覆盖。

## 本地CPU验证

```bash
cd /home/jj/Event-Time-Value-Transport
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_materialize_smolvla_piper_schema6_training_manifest_v2.py \
  tests/test_calibrate_smolvla_piper_adapter_ensemble.py \
  tests/test_evaluate_smolvla_piper_schema6_target_validation50_ensemble.py \
  tests/test_launch_smolvla_piper_schema6_post_collection.py
```

这些测试使用合成 metadata/arrays 和依赖注入，不打开已有 validation/evaluation HDF 或标签，也不产生性能结论。

## 生产 detach 模板

所有 `*_SHA256` 必须在不可变服务器代码目录最终冻结后重新计算；独立 400 对 authority 路径在 detach 时必须不存在。

```bash
python3 scripts/launch_smolvla_piper_schema6_post_collection.py \
  --mode detach \
  --output-root /ABSENT/POST_COLLECTION_ROOT \
  --collection-root /FUTURE/FROZEN/SCHEMA6_COLLECTION_ROOT \
  --source-root /FROZEN/SOURCE63_TRAINING_ROOT \
  --python /BOUND/PYTHON --python-sha256 PYTHON_SHA256 \
  --materializer /IMMUTABLE/materialize_smolvla_piper_schema6_training_manifest_v2.py \
  --materializer-sha256 MATERIALIZER_SHA256 \
  --trainer /IMMUTABLE/train_smolvla_piper_schema6_embodiment_adapter.py \
  --trainer-sha256 TRAINER_SHA256 \
  --target-validation-evaluator /IMMUTABLE/evaluate_smolvla_piper_schema6_target_validation50_ensemble.py \
  --target-validation-evaluator-sha256 EVALUATOR_SHA256 \
  --calibrator /IMMUTABLE/calibrate_smolvla_piper_adapter_ensemble.py \
  --calibrator-sha256 CALIBRATOR_SHA256 \
  --canonical-event-spec /IMMUTABLE/event_spec.json \
  --canonical-event-spec-sha256 EVENT_SPEC_SHA256 \
  --paired-authorization-path /INDEPENDENT/AUTHORITY/paired400.json \
  --gpu-index 0 --gpu-lock /SERVER/LOCKS/etsf_rtx4090_exclusive.lock
```

本次实现未连接远端、未启动生产 watcher、未读取任何现有 fresh/confirmation/validation/evaluation HDF 或标签。
