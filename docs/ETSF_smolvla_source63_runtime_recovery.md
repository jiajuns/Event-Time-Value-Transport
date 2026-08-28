# SmolVLA source63 Torch 运行时漂移恢复

## 事故与修复边界

失败不是数据、初始化状态或共享头结构变化。旧 watcher 将父进程环境完整复制给子进程，只追加
`PYTHONNOUSERSITE=1`，但没有删除 `PYTHONPATH`；同时 `detach` 启动 watcher 时没有显式
`env`。因此同一个显式 Python 路径仍可能在 initializer 与 watcher validator 中导入两套 Torch。
初始化产物记录 `2.4.1+cu121`、validator 当前导入 `2.10.0+cu128`，原有版本一致性检查因而正确
地 fail-closed。不得删除或放松该检查。

修复后的证明链为：

1. 静态预检通过目标解释器的 `-I` 模式探测并冻结 `sys.executable`、Python 版本与 prefix、
   `torch.__version__`、`torch.version.cuda`、`torch.__file__`、`torch._C.__file__` 及两个模块
   文件 SHA；
2. canonical 环境删除所有继承的 `PYTHON*` 变量以及虚拟环境/Conda 路径提示，再显式设置
   `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONHASHSEED=0` 和
   `PYTHONUNBUFFERED=1`；
3. `detach` 用该环境及 `python -I` 启动 watcher。`run` 在创建输出根之前核对自身确实处于同一
   isolated Torch 运行时；
4. initializer 和 trainer 由 `run_etsf_bound_python_stage.py` 在 `-I` 下运行。runner 先导入并
   验证 Torch，随后只把冻结代码根的 `scripts` 目录加入 `sys.path`，再执行获准目标；
5. 每个 stage 前后再次执行 isolated 运行时探针。版本、模块路径、模块字节或解释器任一漂移
   都在新 stage 启动前或终态验证前 fail-closed。

该修复不改变训练数据、split、模型结构、初始化种子、五个训练种子、3000 steps、bf16、
测试标签封存策略或 RTX 4090 独占等待策略。

## 全新目录恢复（服务器执行模板）

旧失败输出必须保持原样且不得复用。先把完整代码依赖部署到一个新的、只读的代码根，并至少确认
下列两个新文件的 SHA 与本地审计值一致：

```bash
NEW_CODE=/home/user/etsf_smolvla_source63_training_code_r7d_20260828
NEW_OUTPUT=/home/user/etsf_smolvla_schema5_native_source_training_r7d_20260828
OLD_OUTPUT=/home/user/etsf_smolvla_schema5_native_source_training_r7c_20260828

test ! -e "$NEW_OUTPUT"
test -r "$OLD_OUTPUT/launch_plan.json"
test -x /home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/bin/python
find "$NEW_CODE" -type d -exec chmod 0555 {} +
find "$NEW_CODE" -type f -exec chmod 0444 {} +
sha256sum \
  "$NEW_CODE/scripts/launch_smolvla_schema5_source63_native_training.py" \
  "$NEW_CODE/scripts/run_etsf_bound_python_stage.py"
```

从旧 plan 只读取完全相同的已登记 collector、split、event spec 和调度参数；输出及代码根必须换新：

```bash
PY=/home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/bin/python
LAUNCHER="$NEW_CODE/scripts/launch_smolvla_schema5_source63_native_training.py"
COLLECTOR=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["collector_root"])' "$OLD_OUTPUT/launch_plan.json")
SPLIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_split"])' "$OLD_OUTPUT/launch_plan.json")
EVENT_SPEC=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["event_spec"])' "$OLD_OUTPUT/launch_plan.json")

"$PY" -I "$LAUNCHER" preflight \
  --code-root "$NEW_CODE" \
  --collector-root "$COLLECTOR" \
  --source-split "$SPLIT" \
  --event-spec "$EVENT_SPEC" \
  --output "$NEW_OUTPUT" \
  --python-bin "$PY" \
  --gpu-index 0 \
  --num-workers 4 \
  --omp-threads 8

"$PY" -I "$LAUNCHER" detach \
  --code-root "$NEW_CODE" \
  --collector-root "$COLLECTOR" \
  --source-split "$SPLIT" \
  --event-spec "$EVENT_SPEC" \
  --output "$NEW_OUTPUT" \
  --python-bin "$PY" \
  --gpu-index 0 \
  --num-workers 4 \
  --omp-threads 8 \
  --detach-receipt "${NEW_OUTPUT}.detach_receipt.json" \
  --detach-log "${NEW_OUTPUT}.launcher.log"
```

若 4090 上已有外部 compute PID，watcher 只更新 `waiting_for_exclusive_idle_rtx4090` 心跳并等待；
不得终止、抢占或修改该进程。collector 已有 `run.exit=0` 时无需重新采集，新的 watcher 会重新认证
完成态元数据、仅自验证 train/validation HDF，并继续保持 test 标签数据集打开计数为 0。

恢复成功至少要求：launch plan、两个 stage receipt 和最终 receipt 中运行时 SHA 一致；initializer
与 validator 的 Torch 版本/模块路径一致；五个成员均完成 3000 steps；输出树最终只读。若任何运行时
身份不同，保留新失败根并再建另一新输出根，不得原地续跑。

## 2026-08-28 r7g 受保护部署

上面的 r7d 命令只保留为历史恢复模板。当前有效 source watcher 是全新、只读的 r7g 代码根：

```text
code   /home/user/etsf_smolvla_source63_training_code_r7g_20260828
output /home/user/etsf_smolvla_schema5_native_source_training_r7g_20260828
pid    1943944（detached，PPID=1，SID=1943944）
```

r7g 不再依赖单次 `nvidia-smi` 空闲判断。静态 plan 冻结并认证用户已有 OpenVLA 全量评测父进程：

- PID `1830377`、start ticks `66143535`；
- boot ID `f0bb2e37-9e9b-4fc8-870e-5e2dddd5793a`；
- raw cmdline SHA `288463ba9c681193366e6ecd5ea473056947eb490c4d6752ac800ff2cca26735`；
- 精确脚本 token `/home/user/openvla-repro/run_all_full.sh` 及文件 SHA
  `fcfae7ad4ce23ba44660f6a836bfce1c7402acf32d3cf4eb8aa9066eeb5f33da`；
- GPU UUID `GPU-06f6e50e-5296-258f-dd86-8f838390a7d1` 与 RTX 4090 D 名称。

每个空闲样本必须满足父进程查询前后都不存在、同一 GPU UUID 上 compute PID 为空；父进程消失后
PID 再出现会失败关闭。只有连续两个有效空闲样本才启动训练。detach 把预检 plan SHA 传给后台
child，child 在创建输出根前必须重算一致。当前三方一致的 static plan SHA 是
`68fa20c5b7286cbbf4f20bf877b85f02b9204ea7d8811e10e79f232d880a68b3`，父进程 guard SHA 是
`01c8760ed2e30d856eb84a5d001e7d5ade13d59003000703feb0259147ef9342`。

部署核验时状态为 `waiting_for_external_suite_parent_exit_before_rtx4090`，外部 OpenVLA 子进程
`1913934` 占用约 15.7 GiB；r7g 没有启动 GPU trainer。旧 r7e watcher 已在确认只完成 CPU
initializer、无训练 stage 后正常 TERM；其文件全部保留。全仓合成/CPU 回归为 `865 passed`。

## 2026-08-28 r7h 最终 fail-closed 部署

r7g 在 GPU training stage 尚未启动时被精确审计并仅对其 watcher PID 执行 TERM；旧代码根、输出根
和收据全部保留，不能复用。它已被新的 r7h 取代：

```text
code   /home/user/etsf_smolvla_source63_training_code_r7h_20260828
output /home/user/etsf_smolvla_schema5_native_source_training_r7h_20260828
pid    1956098（detached，PPID=1，SID=1956098）
lock   /tmp/etsf_smolvla_schema5_source63_r7h_gpu0.lock
```

r7h launcher 文件 SHA 为
`1713fe07a0416ea692cde171061bd739016f4832dc76b0eff7c43904b1c68d57`，远端实现闭包 SHA 为
`0cf9537695fff53be35e0a23899cf96d2438be74289e06a16ecd4d7a29f040e5`。代码根共 15 个普通
文件、3 个目录，无 symlink 和写权限。preflight、detach receipt 与后台 child 物化的 launch plan
三方 static plan SHA 均为
`410c49e452c22354e2d2d90046fad7db08b1ab5b988bb36409138c9919669749`。

第六轮部署审计增加并验证了：Popen 前最后一次 parent/GPU 复核；独立进程组及整组 TERM→KILL→
消失证明；Popen attempted 但生命周期未落盘时保留 GPU 锁；完整 idle/owner/path/token/timeline
语义；terminal 先以 `0000` 创建、树冻结验证后才最后切换为 `0444`；以及独立真实路径 verifier。
定向测试为 `36 passed`，联合 source/bridge/LOBO 为 `131 passed`，全仓为 `884 passed`（仅本地
CUDA 驱动版本探测 warning）。

部署核验时 initializer stage 已 `returncode=0` 且 `process_group_reaped=true`；watcher 状态为
`waiting_for_external_suite_parent_exit_before_rtx4090`。同一受保护父进程 PID `1830377` 仍存活，
其当前官方 OpenVLA 子进程 PID `1951746` 占用约 15.97 GiB。r7h 的 GPU 等待超时为 0（无限等待），
只有父进程消失且同一 GPU UUID 连续两次无 compute PID 后才允许 trainer Popen。当前尚未启动
source63 GPU trainer，也没有读取目标数据或 fresh/confirmation/evaluation 标签。
