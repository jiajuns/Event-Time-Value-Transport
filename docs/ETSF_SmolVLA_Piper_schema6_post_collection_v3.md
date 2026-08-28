# Schema6 post-collection v3（development300）

`scripts/launch_smolvla_piper_schema6_post_collection_v3.py` 是一个全新的、create-once 的后处理编排器。它不复用历史 post v2 的 60/20/50、internal20 calibration 或 paired400 authority 协议。

## 固定数据边界

v3 只接受 materializer v3 的 `development300_v3`：

- adapter train：80 组；
- adapter internal validation：30 组；
- formal target validation：190 组；
- evaluation400：0 组、0 HDF、0 trajectory、0 label。

materializer 实现被固定到 SHA256：

`8a2b4bd4cff0d534e16fe9a97c0c5d52f70f387ea39334e8e4c5bbde8dfa2455`

post v3 校验 aggregate receipt，并逐一校验四个 downstream 文件：target partition、external split、trainer-compatible manifest、expected manifest/split receipt。物化阶段只验证元数据和既有交叉哈希；formal190 内容仍未由 watcher 打开。

它随后复用 trainer 的完整 external split validator，再次证明三集合按原顺序绑定、各自唯一、两两互斥且恰好覆盖 300 组。所有“打开数为 0”的字段必须是 JSON integer `0`；`false` 不会被当作零接受。

## 谱系与五成员训练

启动前的静态计划内容寻址绑定：

- 指定 r7h source root、source final、source plan、source launcher/implementation closure；
- r7h 原生 ensemble manifest 中按固定 seed 排序的五个独立 member checkpoint；
- 实际 r8e final/summary 与 r9b gate final/state/plan 的交叉谱系；
- development300 runner terminal、runner authority、target preregistration、identity authority；
- trainer、formal190 evaluator、calibrator v2、evaluation400 identity bridge v2 和 event spec。

Python、event spec、可选 teacher、全部实现文件以及**实际 import 的 r9b watcher**都必须只读。preregister、run/load 和每次 Popen 紧前都会重新计算文件 SHA；任一文件可写、被替换或内容变化都会 fail closed。

每个 adapter 的 `source-checkpoint` 必须是一一对应的 r7h individual member。LOBO checkpoint 与 aggregate ensemble checkpoint 都不具备训练授权。每个 member receipt v3 使用 exact-field contract，绑定 profile/version、80/30/190、source/checkpoint/summary/internal NPZ SHA、六头 prediction contract、formal190 预开放计数为零，以及 subprocess lifecycle SHA。

只有五个不同 adapter checkpoint 与五个不同 r7h source member 全部冻结且合同一致后，编排器才创建 evaluator input authority。该 authority 把 `target_validation_group_count` 精确写为 190，并继续明确拒绝 fresh、confirmation 和 evaluation400。

在第一份 formal190 authority 创建前，编排器会在 development300 根目录的父目录中以 `O_EXCL|O_NOFOLLOW` 消耗一份由 development300 terminal identity 派生的全局 one-shot claim。同一 terminal 即使在 evaluator 或 calibrator 阶段失败，也不能换一个 output root 再次开放 formal190；failure receipt 会明确绑定 claim 是否已消耗以及已经存在的 evaluator receipt SHA（若可验证）。

## 六头校准与 bridge handoff

独立 evaluator 是唯一获准打开 formal190 的进程，并输出一份 common labels NPZ、五份 member prediction NPZ 和 calibrator v2 input authority。calibrator v2 对以下六头做支持、校准与不确定性合同验证：

- post event；
- next event（仅 `duration_observed`，不把结构性 e0 或 censor self-loop 当监督）；
- duration（`log1p` shifted-lognormal）；
- success；
- conditional recovery（`p(recovery | operational regress)`，stop-gradient，五 member 均训练且正负 group 支持独立过门后才可进入 primary）；
- object effect（物理 delta 空间）。

post v3 只有在 post/next/duration/success/recovery/object **六头全部启用**，且 abstention threshold 通过 bridge v2 的现有 validator 后才发布 handoff。evaluator receipt 必须逐项绑定刚生成的 input authority；calibration authority 和 calibration receipt 必须沿用 evaluator 给出的同一 path/file/logical identity，且 `validation_groups` 必须是严格 integer `190`。handoff 内容寻址绑定本流水线产生的四项 bridge 依赖：

- ensemble manifest；
- calibration；
- head support；
- calibration receipt。

target manifest、private identity disjoint attestation 和 policy bridge receipt 仍必须由外部 pre-outcome 流程提供。本 handoff 本身不执行 identity bridge、不授权 evaluation400 执行，也不等待或生成旧 paired400 authority；`second_reserve400_created=false`。

## 进程与终态

每个 subprocess 都使用 `start_new_session=True`，并强制证明 `PGID == PID`。异常路径对可信的独立进程组执行 TERM→KILL，随后同时证明 direct process 与整个 process group 已回收。若 Popen 已尝试但 PGID/整组回收无法证明，输出保持可写且不发布 terminal，shared GPU lock 在常驻 watcher 中持续保留。

GPU lock 通过 `O_NOFOLLOW` 的安全 fd 获取。父环境不得预设非空 `CUDA_VISIBLE_DEVICES`；两次 idle audit 固定同一 4090 的 index/name/UUID，GPU 子进程只暴露该 UUID 并统一使用 `cuda:0`，Popen 紧前再次核对 UUID。CPU stage 在尚未获取 GPU lock 时生命周期不明，不会虚报 `gpu_lock_retained=true`。detach 若无法证明 `PGID==PID`，只终止并等待 direct child，绝不向未知进程组发送信号。

成功终态先把 `final_receipt.json` 与 `run.exit` 以 mode `000` 创建，再冻结并验证整棵输出树，先发布只读 `run.exit`，最后才发布只读 `final_receipt.json`。冻结失败按 inode 删除隐藏终态，不能留下可接受 success。

终态的 artifact closure 会真实重开并重哈希 static plan、PPID1 detach proof、两份 GPU idle audit、GPU release、materializer/evaluator/calibrator receipt、五份 member receipt、handoff、全局 claim，以及八个 stage 的 launch/lifecycle/log/exit；任一闭包文件变化都会使 terminal validator 拒绝发布。

`detach` 使用独立 session 启动 worker；worker 在读取任何上游输入前等待并写入 PPID 1 proof。计划、输出和所有 stage 都是 create-once/non-resumable，避免把失败残留误当成可恢复成功。

## 合成 CPU 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  python3 -m pytest -q \
  tests/test_launch_smolvla_piper_schema6_post_collection_v3.py
```

测试只构造 JSON、普通字节 checkpoint/NPZ 占位和短生命周期 CPU 子进程；不读取真实 HDF、trajectory 或 label。
