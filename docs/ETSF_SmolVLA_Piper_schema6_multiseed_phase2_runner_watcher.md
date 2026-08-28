# SmolVLA/Piper Schema6 多 seed Phase-2 runner 与 watcher

## 当前结论

本次新增最小生产 runner、detached watcher、真实可导入 runtime/reset adapter 与离线 authority freezer；没有修改冻结 r6j 文件、构造真实环境、加载真实 policy，或读取 evaluation/Fresh/confirmation/test 数据。4090 服务器上的完整 200-decision runtime contract 已离线冻结，但当前仍不能安全部署生产采集：CPU preregistration 自身明确 `production_execution_authorized=false`，真实 target manifest、private-disjoint attestation 和独立 execution authority 尚未冻结。`preflight` 在这些依赖缺失时会在创建 output root 和构造环境之前 fail closed。

服务器曾先冻结一个 `max_episode_steps=4` 的 v2a contract。它只足以做接口 smoke，不能覆盖抓取、搬运、放置、失败和恢复事件，因此禁止用于正式 130-group 数据。权威候选是随后 create-once 冻结的 full-horizon v2b：

- path: `/home/user/etsf_smolvla_piper_schema6_runtime_contract_v2b_20260828/runtime_contract.json`
- file SHA256: `4552759f13bebea17a9a9097b7d444c0fd3547d809737390559d901b8bfa06c5`
- logical SHA256: `90365b2467c44b3b9b79b7af7f26c92b68c4ef1646ebcc59ec064432aa611a7b`
- `max_episode_steps=200`
- measured channel: `task.robot.get_left_arm_real_jointState+get_right_arm_real_jointState`

## 新增文件

- `scripts/run_smolvla_piper_schema6_multiseed_v2.py`：逐 seed runner、reset/registry/pose 复核、四候选 accounting、create-once group/receipt。
- `scripts/launch_smolvla_piper_schema6_multiseed_v2.py`：签名静态计划、gap-free prefix 恢复、RTX4090 全程独占锁、detached/PPID1 和顺序 130-command watcher。
- `tests/test_smolvla_piper_schema6_multiseed_v2_runtime.py`：纯 CPU dependency-injection fake runtime/collector 测试。
- `scripts/smolvla_piper_schema6_runtime_adapter_v2.py`：复用 RoboTwin/SmolVLA/Piper 链的 `build_runtime`，以及共享同一 reset/identity reader、禁止 policy 加载的 `build_reset_only_adapter`。
- `scripts/freeze_smolvla_piper_schema6_runtime_authorities_v2.py`：离线冻结 runtime contract、reset-only authority 和 Phase-2 authority。
- `tests/test_smolvla_piper_schema6_runtime_adapter_v2.py`：纯 CPU factory/interface 与隐式 seed retry 不入选、继续候选测试。

## Runner 数据路径

runner 消费 v2 preregistration 的严格签名和 130 条 command。每条 command 必须保持 adaptation80 后 validation50 的固定 manifest 顺序；不会生成或执行 evaluation command。

每 seed 的流程为：

1. create-once 创建该 seed root；旧目录、partial 或 symlink 都拒绝覆盖。
2. 用 requested seed 和固定 instruction reset，核对 resolved seed。
3. 从 runtime identity snapshot 重新计算并核对：
   - initial scene SHA；
   - measured 14-d joint-state SHA；
   - commanded 14-d drive-target SHA。
4. 从 live task 动态调用 r6j materializer 构造 can/pot registry，随后生成绑定该 registry SHA 的 pose-quality spec；两者均 create-once 写入 seed root。
5. 在任何 policy query 前写签名 `per_seed_reset_receipt.json`。
6. 使用绑定 SHA 的 r6j dense collector 执行所有合法根候选；其每次 branch reset 都经过 runner wrapper，再次核对 resolved/initial hashes 和 live registry identity。
7. 对原始 `[0,1,2,3]` 创建严格四行 accounting：合法候选必须存在 executed r6j branch；非法候选必须不存在 branch，且仅记录 `nonexecuted_censored_infeasible`、`executed=false`、`right_censored=true`。
8. r6j legal-branch HDF 先写到 staging 并通过 r6j validator，再追加独立 `candidate_accounting_v2`；v2 validator 通过后用 hard-link create-once 发布最终 `schema6_group.hdf5`。
9. 最后写签名 `completed_group_receipt.json`，绑定 prereg/command/reset/registry/pose/group SHA 和精确四分支计数。

r6j collector 若没有得到完整合法分支 group（例如少于其要求的可行候选数）会 fail closed，不会发布伪完成 receipt。

## Production execution authority

preregistration 是 CPU 计划，不是执行授权。生产 runner 还要求环境变量 `ETSF_SCHEMA6_V2_EXECUTION_AUTHORITY` 指向只读签名 authority：

```text
format = etsf_smolvla_piper_schema6_multiseed_execution_authority_v2
status = authorized_adaptation80_validation50_collection_only
production_execution_authorized = true
```

authority 必须绑定 preregistration 路径、文件 SHA、逻辑 SHA、runner 路径/SHA、runtime adapter 路径/SHA、`move_can_pot.py` 路径/SHA，并嵌入签名 runtime contract。该 contract 绑定 RLinf/RoboTwin/LeRobot 根、五个 simulator source、eval seed registry、SmolVLA/VLM 全目录树 SHA、14D action bounds、measured-state channel、reset scratch、GPU0/RTX4090 和 offline-only 范围；明确 evaluation command 数为 0、test 未读、Fresh/confirmation 不接受。runner 在 authority 完整通过前不会 import adapter 或构造环境。

此外 preflight 会逐字节重验 target manifest、r6j 七个 runtime artifacts 及 closure。production adapter 必须暴露：

```python
build_runtime(command=command, event_spec=event_spec) -> {
    "runtime": {...},
    "query_fn": callable,
    "max_steps": int,
    "close": callable,
}
```

runtime 至少提供 `reset`、`identity_snapshot`、`task`，以及 r6j collector 所需的 `snapshot/step/derive_events`。

## Watcher 与恢复

`detach` 只有在完整 production preflight 成功后才 create-once 创建 collection root、`_watcher/static_plan.json` 和原子 state，再以 `start_new_session=True` 启动。server process 在取得 GPU lock 前无限等待 `PPID=1`，因此 SSH/本地电脑退出不会结束任务。

watcher 在全部 130 commands 的整个窗口持有同一非阻塞 `flock`，并要求设备名包含 `RTX 4090`。开始前与全部结束后都要求连续两次 compute-PID 为空；没有总 timeout。每 seed 子进程运行时，GPU compute PID 必须属于该 runner 的进程树，否则 watcher 终止进程组并 fail closed。每 stage 都 create-once 写 launch receipt、`run.log`、`run.exit` 和 stage receipt。

恢复只接受从第 0 条 command 起的连续签名 completed receipt 前缀，并重新计算每个 HDF 字节 SHA、reset receipt 逻辑 SHA。首个未完成 command 已存在 partial seed root、任何后续目录/receipt、签名漂移或文件 SHA 漂移都会拒绝恢复。`detach-resume` 的 receipt 名称由 completed-prefix 长度确定，同一 prefix 不能重复启动。

完整 130-group prefix 后输出终态 receipt、严格 `run.exit=0\n` 并递归冻结只读。明确捕获的失败也写失败 receipt、`run.exit=1\n` 并冻结审计现场。

## CPU 测试

```bash
cd /home/jj/Event-Time-Value-Transport
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_smolvla_piper_target_seed_manifest.py \
  tests/test_smolvla_piper_schema6_runtime_adapter_v2.py \
  tests/test_smolvla_piper_schema6_multiseed_v2_runtime.py
```

测试使用 fake identity/runtime/collector，不访问 GPU 或真实数据。覆盖：两个 factory 的精确接口、RoboTwin 左/右 real jointState 拼接 measured 14D、measured/commanded identity 分离、reset-only 不加载 policy、隐式 seed retry 只记 `unstable` 且不暴露/选择 retry identity、runtime contract authority hard gate、五次 reset 均复核 identity、live registry branch-reset 复核、`[executed,censored,executed,executed]` 四候选 HDF accounting、identity 漂移早停、later partial/gap 拒绝、authority 缺失时 output 未创建、两次 idle、PPID1 以及非法候选被伪报 executed 时拒绝。

## 未来生产命令边界

只有 target manifest、r6j closure、runner/runtime adapter、event spec、move task source 与 execution authority 全部冻结且 SHA 匹配后，才允许先运行：

```bash
python3 scripts/launch_smolvla_piper_schema6_multiseed_v2.py \
  --mode preflight \
  --preregistration /IMMUTABLE/multiseed_preregistration.json \
  --preregistration-file-sha256 PREREG_FILE_SHA256 \
  --execution-authority /IMMUTABLE/execution_authority.json \
  --output-root /ABSENT/phase2_collection_root \
  --gpu-index 0 \
  --gpu-lock /SERVER/LOCKS/rtx4090.lock
```

preflight 成功后才可将 `--mode preflight` 改为 `--mode detach`。当前前置条件不满足，因此本次没有提供或运行任何远端启动命令实例。
