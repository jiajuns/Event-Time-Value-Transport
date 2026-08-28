# SmolVLA–Piper schema-v6 远端自主 watcher

入口：`scripts/launch_smolvla_piper_schema6_autonomous_watcher.py`。

该 watcher 只负责服务器端顺序编排，不在用户本机运行训练，也不把一次开发采集解释为性能或迁移证据。它采用 create-once 输出、原子状态、内容哈希、进程互斥锁和 detached 新会话，SSH 断开或用户本机关机不会中止服务器端进程。

## 固定执行顺序

1. 等待固定 LOBO aggregate 根目录完成；复核总 `run.exit` 和 Piper、UR5 两个子 `run.exit` 均为严格字节 `0\n`，两个 stage receipt、进程生命周期证明与 state/final 完全一致，外部 `lobo_training_summary.json` SHA/本体/零目标训练载荷/零 test HDF 契约均成立。除 LOBO 的传递证明外，watcher 还直接重绑 r7h source 根、launch-plan 文件 SHA、static-plan SHA、launcher SHA、implementation closure SHA、五成员 checkpoint 实际字节和原生 deployment checkpoint，防止谱系在下游丢失或被替换。
2. 取得 schema6 专用 create-once 锁，并确认物理 GPU 0 的名称包含 `RTX 4090`、compute-app 数为 0。
3. 调用已部署只读 r6j 代码的 reset-only materializer。该步骤只 reset 一次，保存 can/pot 的 task attribute、SAPIEN actor name 和资产 model id；禁止 policy forward、`env.step`、轨迹或标签读取。
4. 用 materializer 的两个只读 JSON、冻结 event spec 和 R6f lineage 生成新的内容寻址 authority。authority 固定一个非敏感开发 seed、四个候选、root/continuation 动作 horizon 均为 1。
5. 再次确认 4090 无计算进程，然后由 authority 启动一次 schema6 collection。

LOBO gate 在取得 schema6 GPU 锁之前完成，因而不会在 LOBO 训练期间抢占 4090。多个 watcher 即使使用不同输出根，也只能有一个取得全局 schema6 锁；锁残留会 fail-closed，不会自动猜测或删除。

## 路径与数据边界

- LOBO aggregate 根固定为 `/home/user/etsf_multibody_lobo_autonomous_r8e_20260828`。
- 两个 LOBO 外部 summary 固定属于 Piper 与 UR5 的 r8e 输出根。
- source 根固定为 `/home/user/etsf_smolvla_schema5_native_source_training_r7h_20260828`，并精确绑定 r7h 的 plan、static plan、launcher 与实现闭包内容哈希；LOBO checkpoint 仅用于跨本体科学评估，部署动作重排权限始终只属于 r7h 原生五成员 ensemble。
- schema6 实现根固定为 `/home/user/etsf_smolvla_piper_schema6_code_r6j_20260828`，执行前和每个阶段前都重算递归本地-import closure SHA。此前未通过 preflight 的 r6g/r6h/r6i watcher 不作为执行证据，已冻结目录不修改。
- event spec 延续 Stage1 的 canonical 语义：`anchor=""` 与 `anchor=null` 都表示绝对目标中心、没有相对 anchor；`anchor="pot"` 表示相对 pot。其他 anchor 值仍 fail-closed。
- Python 固定为 SmolVLA RoboTwin 评测环境；GPU 索引固定为 0。
- 每个子进程都从父环境删除 `PYTHONPATH`、`PYTHONHOME`、virtualenv/conda 元数据，再强制 `PYTHONNOUSERSITE=1` 等确定性变量；清理契约 SHA 同时写入 state、各 stage receipt 和最终 receipt，避免父 shell 把另一套 Torch 注入指定 venv。
- 任一路径在词法形式或解析后的任一 component 命中敏感命名空间都会在任何文件打开之前被拒绝；JSON 中嵌入的路径也会审计。全局 `load_json` 规则没有放宽。
- 旧 R6e/R6f 的 canonical lineage 在 `inherited_R6e_contract.development_seed.path` 精确位置携带一个历史敏感路径字符串。r6i 只在 R6f logical SHA、inherited-contract SHA、R6e logical SHA、两份 seed record 完全相同且固定 seed/label-free 标量成立时，把该字符串解释为签名 lineage 元数据；只计算其 UTF-8 SHA256，并保留旧 manifest 内容 SHA，不调用 `Path`、`resolve`、`stat`、`open`，也不把原字符串写入 plan/state/receipt。
- materializer/freezer/collector 共用的 lineage loader 不再调用会解引用旧 seed manifest 的 R6e `_load_and_recompute_preregistration`。它仍重绑 R6c/R6d、重哈希 runtime source/model/VLM，随后以 `etsf_signed_legacy_seed_lineage_projection_v1` 的无路径投影替换旧字段。除该精确签名位置外出现任意第二个敏感路径字符串都会 fail-closed。
- 历史 R6e 的 `direct_actor_runner`、`r6d_base_executor`、`shared_prefix_capture` 路径来自签名时的旧只读代码根，而重建器会自然产生当前 r6j 的 `__file__` 路径。r6j 只规范化这三个路径字段及其派生 logical SHA：旧文件必须仍是 canonical、非 symlink、只读 regular file，记录 SHA 必须同时等于旧文件实际字节 SHA 和当前重建实现 SHA。规范化后执行整份 R6e exact comparison；出现第四个差异或任一非路径差异均 fail-closed。
- watcher 不导入 HDF5 库，不打开任何 source/test HDF。它只读取 LOBO aggregate 及其外部 summary，并按字节哈希日志和 JSON。唯一允许读取的 HDF 字节是本 watcher 刚生成的非敏感开发 group，用于核对 collection receipt SHA。

## 启动方式

先在 4090 服务器上把本脚本部署为一个新的只读文件并记录 SHA。R6f preregistration、event spec 必须已经冻结只读，watcher 输出、detach receipt 和日志必须不存在。

静态预检不会读取 LOBO 终态 summary，也不会创建 watcher 输出：

```bash
python launch_smolvla_piper_schema6_autonomous_watcher.py preflight \
  --r6f-preregistration /absolute/frozen/r6f.json \
  --event-spec /absolute/frozen/event_spec.json \
  --output /home/user/etsf_smolvla_piper_schema6_autonomous_r9b_20260828
```

通过后 detached 启动：

```bash
python launch_smolvla_piper_schema6_autonomous_watcher.py detach \
  --r6f-preregistration /absolute/frozen/r6f.json \
  --event-spec /absolute/frozen/event_spec.json \
  --output /home/user/etsf_smolvla_piper_schema6_autonomous_r9b_20260828
```

`detach` 使用 `start_new_session=True`、stdin `/dev/null`、关闭继承 fd，并原子发布只读 detach receipt。收到 receipt 后用户本机可以断开，但仍应由服务器端监控最终 `launch_state.json` 或 `final_receipt.json`。

## 终态语义

三个本 watcher 阶段都以 `start_new_session=True` 进入独立进程组，并保存 argv SHA、允许的 return code、真实 return code、PID/PGID、父进程与整组回收证明、日志 SHA 和产物审计 SHA。超时、异常或正常退出后遗留孙进程都会触发 TERM→KILL 整组清理；只要 Popen 已尝试但回收证明不完整，GPU 锁就不会释放。materializer/freezer 只接受 0；collection 接受：

- `0`：完成一个 schema6 开发 group；
- `20`：root 少于两个合法候选，完成一次 fail-closed 空采集尝试。

`20` 不是模型或环境故障，也不是成功率证据；`21` 或任何未登记退出码均使总 watcher 失败。成功或可安全发布的失败终态先以 mode `000` 创建 `run.exit` 和终态 receipt，冻结并逐项验证整棵树后，先发布只读 `run.exit`，最后发布只读 receipt；冻结失败会按 device/inode 删除隐藏终态，避免外部消费者看到半冻结的伪完成结果。成功收据明确保留 `performance_or_transfer_claim_authorized=false`。后续仍需足量 schema6 数据、目标 adapter 和成对任务成功率实验才能支持迁移/改善结论。

## 2026-08-28 r9b 部署证据

历史 r9 使用 watcher v1 绑定了已失败的 r8，因此按协议冻结为 failure；该输出不复用。v2 部署使用全新的 r9b 根：

- watcher 代码根：`/home/user/etsf_smolvla_piper_schema6_watcher_code_r9b_20260828`
- watcher SHA256：`40916a07ddcd98706ac90eecf5ff86709cc73123f3f58a77d3e8ddb341887428`
- 输出根：`/home/user/etsf_smolvla_piper_schema6_autonomous_r9b_20260828`
- static plan SHA256：`5d45cc8e08a942c31105e802c16d9ff60bf13d0bef5e0e3e52bfd8b66172d805`
- launch plan file SHA256：`39bfada50a5e59dfec6a4d1830a0f4d18fb0b307d4ba83e4a32dc85901a13fa2`
- detached PID：`1965014`，PPID `1`，SID `1965014`
- process start ticks：`71134245`；boot ID：`f0bb2e37-9e9b-4fc8-870e-5e2dddd5793a`
- raw cmdline SHA256：`597a6873b1568e3a39946beeb8621d42f127366e4b846028e10aec06ac846f76`
- detach logical SHA256：`af39f8a800cc5dafead4647555e77c99542ccb3193d916432c91f8c335ef9839`
- R6f file SHA256：`0828d39b2aff71ec48b500b47a08fd30954950a4a437417e38c5828865626ba9`
- event spec SHA256：`8b1ff070ee7f9519707f45c42209b266a515b45c95d14f13bb4a95f283f2bff5`

启动后的权威 state 为 `waiting_for_piper_then_ur5_lobo_terminal_no_hdf5_access`，`stage_lifecycles=[]`、`stages_started=[]`，Schema6 GPU 锁不存在；因此它尚未启动任何采集子进程，也未占用 4090。它会在 r8e 发布通过完整谱系验证的 frozen final 后才尝试取得 Schema6 GPU 锁。
