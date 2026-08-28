# SmolVLA/Piper baseline↔事件插件成对任务成功率验证

## 当前交付边界

本实现只冻结非敏感 development 协议并提供 CPU synthetic tests；未连接服务器、未启动远端 GPU、未读取任何已有敏感命名空间或 sealed reserve 身份/结果。只有 LOBO、Schema6 和 adapter 三个签名终态均可验证后，才能生成生产协议并启动 development watcher。Schema6 的权威依赖根必须来自 `/home/user/etsf_smolvla_piper_schema6_autonomous_r9_20260828`；旧 r8 不能接受。adapter 必须使用后续冻结的 aggregate receipt 契约，不能仅凭 trainer 的 `training_summary.json` 猜测成功。

## 主比较是什么

每个成对单位固定同一个 Piper `move_can_pot` 任务、requested/resolved seed、reset identity、SmolVLA actor checkpoint、根观测、四个根动作候选、候选噪声和根动作后的 lowest-legal continuation。baseline 选 lowest-legal 根候选；插件只在第一个环境 step 之前读取相同根候选，并用 `O_EXCL` 写出不可覆盖的选择记录。现有 Schema6 dense branch collector 随后从同一 reset 执行每个可行根分支。主结果来自已执行分支的 `simulator info.success`，预测 success/logit、事件准确率或不确定性指标都不能替代任务成功率。

插件的主效用是 actor-agnostic structured utility：

```text
z(E[next event progress]) - z(E[post event progress]) + z(duration log mean)
+ 0.5 z(success probability) + 0.5 z(object-effect utility)
```

post-event、next-event 与 duration core head 的冻结 support minimum 为每侧 10 个独立 group；success 与 object-effect optional 项保持每侧至少 50。每个 head 的 receipt 必须记录对应 minimum，validator 会逐 head 精确复核；支持不足就强制关闭。aleatoric 与 epistemic uncertainty 的和只作 guard，不进入成功结果或 utility；非有限或超过预注册阈值时退回 baseline。post/next/duration/success/object 各 head ablation 与 success-only 都是 secondary diagnostics，绝不改变主统计门。

## 主统计门

主 estimand 是所有预注册 development pair 上无条件的 `plugin task success - baseline task success`；abstain 作为与 baseline 相同的 tie 保留，不能筛掉。当前生产门同时要求：

- 预注册且完整 pair 至少 400，全部 pair 通过 intention-to-treat 完整性检查；
- executed policy changes 至少 40、覆盖率至少 10%；
- discordant pairs 至少 20；
- 成对 delta 严格大于 0；
- 固定 seed、20,000 次 paired bootstrap 的 95% CI 下界严格大于 0；
- exact two-sided McNemar/sign test `p < 0.05`；
- executed changes 中 harmful rate 不高于 10%。

实验进程返回 0 但统计门未通过是一个有效的科学零结果，终态为 `complete_development_task_success_gate_not_passed`；它不会伪装成执行故障，也不能支持“提高成功率”结论。只有 `complete_development_task_success_gate_passed` 才支持当前固定 Piper development 范围内的成功率提升证据，仍不等于 sealed 验证或跨本体结论。

## 依赖与防泄漏契约

`dependency_authority.json` 必须按 `lobo → schema6 → adapter` 顺序逐项绑定：终态 receipt 路径、文件 SHA、format/status、逻辑 SHA 字段、关键 required fields，以及内容严格为 `0\n` 的 `run.exit` 路径和 SHA。协议冻结与 watcher 执行前后都会重验这些字段。seed authority 只公开 development reset identities；sealed reserve 只允许 count 与 identity-set SHA，身份和 outcome 都不可见、不可执行。

所有输入、输出和嵌入路径只要任一 component 包含被禁用的敏感 token 就 fail closed。watcher 不打开 HDF5；只读取冻结协议、依赖 JSON、pre-outcome selection JSON 和实际分支 pair-result JSON。

## 本地验证

```bash
cd /home/jj/Event-Time-Value-Transport
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_smolvla_piper_paired_success_protocol.py \
  tests/test_launch_smolvla_piper_paired_success_development.py
python3 scripts/smolvla_piper_paired_success_protocol.py --mode synthetic-smoke
```

synthetic smoke 只验证统计/编排语义，不是模型结果，也不能用于性能声明。

## 生产冻结与守护模板

adapter aggregate receipt 定稿且三项依赖全部完成后，先用 `freeze-protocol` 生成一个全新协议文件，显式绑定 seed/dependency authority、插件 manifest、adapter checkpoint、Schema6 collector、事件插件、structured utility 和 head-support receipt。随后在 RTX 4090 服务器上使用全新 output root：

```bash
python3 scripts/launch_smolvla_piper_paired_success_development.py \
  --mode detach \
  --output-root /ABSENT/PAIRED_DEVELOPMENT_ROOT \
  --protocol /IMMUTABLE/paired_protocol.json \
  --protocol-file-sha256 PROTOCOL_FILE_SHA256 \
  --executor /IMMUTABLE/paired_executor.py \
  --executor-sha256 EXECUTOR_SHA256 \
  --python /MATERIALIZED/NON_SYMLINK/PYTHON \
  --gpu-index 0 \
  --lock-path /SERVER/LOCKS/etsf_rtx4090_exclusive.lock
```

守护器在 detach 之前创建 output root、collection/stage 目录、签名 static plan 与原子 state，随后以新 session 启动。执行前要求两次连续 RTX 4090 compute-PID 空闲检查并持有非阻塞独占锁；运行中出现非 executor 后代 GPU PID 会终止；结束后再次要求两次连续空闲。stage 总是绑定 `run.log`、`run.exit` 和 `stage_receipt.json`；成功后发布 `paired_success_evaluation.json`、`final_receipt.json`、根 `run.exit=0\n` 并将树冻结为只读。

当前仓库只规定 executor 的严格接口与 collection manifest 契约，没有伪造一个尚未绑定真实 adapter aggregate receipt 的远端执行实例，因此现在不应运行上述生产模板。
