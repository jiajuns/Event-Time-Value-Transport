# Schema6 development300 exact collection runner

本协议是 development300 identity authority 之后的独立执行层。它不会扩大 identity preregistration 的能力，也不会生成、读取或执行 evaluation400。实现由两个新程序组成：

- `run_smolvla_piper_schema6_development300_collection.py`：物化 runner authority、生成 dry-run exact plan、认领一次性输出根、detach 并监控 300 个命令；
- `execute_smolvla_piper_schema6_development300_group.py`：每组独立 sealed worker。只有这个子进程能接触 collector 返回的 label-bearing 内存记录；它写盘后只向 runner 发布 outcome-free receipt。

## Authority 输入和能力边界

runner authority 同时重新验证并绑定：

- `collection_identity_authority.json` 的文件 SHA 和逻辑 SHA；
- `collection_preregistration.json` 的文件 SHA、逻辑 SHA、300 个 command SHA 及顺序；
- full-horizon 200-step runtime contract v2b 的文件 SHA、逻辑 SHA、模型树、VLM 树和 runtime source artifacts；
- dense collector、runtime adapter、sealed worker、runner、Python 解释器及本地传递依赖闭包的文件 SHA；
- event specification 文件 SHA；
- 一个尚不存在的、与 preregistration 完全相同的 output root；
- GPU 0、RTX 4090 身份、独占 lock 路径和启动前两次空闲观测。

authority 固定 train80 / internal30 / formal190 的原始顺序，不允许失败重试、seed 替换、额外 seed、结果依赖停止、失败恢复或重入。它不授权 formal190 label open、checkpoint selection、fresh/confirmation 输入或 evaluation400。

## 物化 authority

所有路径必须是服务器上的最终路径，SHA 必须在代码和输入完成同步后重新计算。authority 会绑定 runner/worker 自身 SHA，因此之后修改任一实现都会使 preflight 失败。

```bash
python scripts/run_smolvla_piper_schema6_development300_collection.py \
  materialize-authority \
  --identity-authority COLLECTION_IDENTITY_AUTHORITY.json \
  --identity-authority-file-sha256 IDENTITY_FILE_SHA \
  --identity-authority-sha256 IDENTITY_LOGICAL_SHA \
  --collection-preregistration COLLECTION_PREREGISTRATION.json \
  --collection-preregistration-file-sha256 COLLECTION_FILE_SHA \
  --collection-preregistration-sha256 COLLECTION_LOGICAL_SHA \
  --runtime-contract RUNTIME_V2B.json \
  --runtime-contract-file-sha256 RUNTIME_FILE_SHA \
  --runtime-contract-sha256 RUNTIME_LOGICAL_SHA \
  --collector scripts/collect_smolvla_piper_schema6_dense_event_branches.py \
  --collector-file-sha256 COLLECTOR_SHA \
  --runtime-adapter scripts/smolvla_piper_schema6_runtime_adapter_v2.py \
  --runtime-adapter-file-sha256 ADAPTER_SHA \
  --sealed-worker scripts/execute_smolvla_piper_schema6_development300_group.py \
  --sealed-worker-file-sha256 WORKER_SHA \
  --event-spec EVENT_SPEC.json \
  --event-spec-file-sha256 EVENT_SPEC_SHA \
  --gpu-lock /SERVER/LOCKS/schema6-development300-gpu0.lock \
  --output development300_collection_runner_authority.json
```

该命令只读取身份/预注册/runtime/代码/事件规范的静态字节，不构造环境、不调用策略、不创建 output root，也不打开任何 HDF。

## Dry-run exact plan

在 detach 前必须先运行 dry-run：

```bash
python scripts/run_smolvla_piper_schema6_development300_collection.py \
  dry-run-plan \
  --authority development300_collection_runner_authority.json \
  --authority-file-sha256 RUNNER_AUTHORITY_FILE_SHA
```

成功输出必须包含 300 个唯一 command SHA、固定 order SHA、1200 个候选记账位置、精确 80/30/190 和 `evaluation400_commands=0`。此时 output root 仍必须不存在。

## Detached server-side 执行

确认 dry-run 后才能执行：

```bash
python scripts/run_smolvla_piper_schema6_development300_collection.py \
  detach \
  --authority development300_collection_runner_authority.json \
  --authority-file-sha256 RUNNER_AUTHORITY_FILE_SHA \
  --idle-interval-seconds 30
```

detach 以 `start_new_session=True`、关闭 stdin 和继承 fd 的方式启动服务器进程。子进程必须先变成 PPID 1，随后以 `O_EXCL` 写入一次性 run claim、获取 GPU lock、确认两次 RTX 4090 空闲，再按 command 0..299 顺序启动 sealed worker。协议没有 resume 子命令；相同 output root 无法再次 detach 或 serve。

每个 stage 在启动前固化 worker argv、authority file SHA、static plan file SHA、global ordinal 和 command SHA。worker 只能处理当前 preregistered requested/resolved identity，运行时每次 reset 都重新验证 scene/qpos/drive identity；不得隐式 seed retry，也不能生成替代命令。

## 四候选语义

每组固定产生 original candidate index `[0,1,2,3]` 的四条记账记录，共 1200 条。冻结 collector 的合法性语义保持不变：合法候选执行分支；根部不可行候选记录为 `nonexecuted_censored_infeasible`，不得强行送入环境，也不得用其他候选或 seed 替换。如果一个根少于两个合法候选，现有 collector 不会产生完整 group；exact runner 会在该组失败关闭，且不重试、不替换、不继续。

因此“300×4”在本协议中精确定义为 1200 条冻结候选身份与 executed/censored 完整记账，不虚构“必然有 1200 条环境执行轨迹”。如果后续研究要求四个候选无论可行性都实际 step，需要新的 collector 版本和新 authority，不能重解释本协议。

## Formal190 label isolation

collector 的原接口会返回包含 success、trajectory success 和 event 的内存字典。为了不让 runner/watcher 打开这些标签，worker 在独立进程中完成以下操作：

1. 在任何 policy query 前写 identity-only reset receipt；
2. 运行 frozen collector，并在 worker 内保存和结构验证 HDF；
3. 生成不含 success/event/outcome/label 的四候选 accounting；
4. 生成 outcome-free completed receipt；
5. 对 formal group 将全部文件改为 `0400`、目录改为 `0500`，然后退出；
6. runner 只验证 JSON receipt、SHA 和文件权限，将 staging 目录原子移动到预注册 final path。runner 对 HDF 只做不解释内容的字节 SHA，不解析 HDF group/dataset。

train80/internal30 输出也在发布后只读，但不带 formal label-open 授权。formal190 即使已经采集完成，仍需要单独的 label-open/evaluator authority；runner terminal receipt 不能作为标签授权。

## Failure receipts 与不可恢复性

每个 worker 只有一次调用机会。非零退出、输出缺失、SHA 不符、candidate accounting 不完整、身份变化、已有 final path 或任何验证异常都会：

- 立即停止，不启动下一命令；
- 将残留 staging payload 设为只读；formal stage 使用 `0400/0500`；
- 原子写入 stage failure receipt 和 terminal failure receipt；
- 冻结整个 output root；
- 明确写入 `retry_or_resume_authorized=false`。

进程在第一组前失败时也会写 sanitized pre-loop terminal receipt。无法捕获的 `SIGKILL`、主机掉电或文件系统失效不可能保证写出终态收据；此时一次性 run claim/launch receipt 仍使原 root fail-closed，协议禁止原地恢复。继续只能新建版本化 authority 和全新 output root，不能伪造 terminal success。

## 本地合成验证

测试只构造临时 runtime 文件、身份摘要、opaque synthetic HDF bytes 和 outcome-free receipts，不连接远端、不执行模拟器/策略，也不读取任何真实 fresh、confirmation、formal、evaluation、trajectory、label 或 HDF。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jj/miniconda3/envs/spatial-lite-local/bin/python -m pytest -q \
  tests/test_run_smolvla_piper_schema6_development300_collection.py
```

测试覆盖 authority 全量重算、dry-run 300-command plan、实现/命令篡改、output root 二次认领、detached command、sealed worker 标签净化、完整 300 组一次发布、formal190 权限封存，以及第 6 个 worker 失败后只保留 5 组精确前缀且绝不重试。
