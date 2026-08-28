# ETSF 多本体 LOBO 自主顺序 watcher

入口：`scripts/launch_multibody_lobo_autonomous.py`。

该程序只做服务器端编排，不导入训练器、不打开 HDF5，也不修改 LOBO 模型代码。它将已经
冻结的 Piper 与 UR5 leave-one-body-out 实验注册成一个不可恢复重入的顺序任务：

1. 等待既有 SmolVLA source63 watcher 的 `final_receipt.json`；
2. 核对 source63 的冻结 `launch_plan.json`、五成员各 3000 step 训练审计、目标/测试标签读取为
   0，并要求整个上游输出树已经只读；
3. 获取与 source63 相同的 GPU lock，并连续两次确认 GPU 0 是 RTX 4090 且没有 compute PID；
4. 先运行冻结 Piper LOBO；验证其十个被选 checkpoint、终评 summary 与 test-HDF=0 后冻结输出；
5. 再次确认 GPU 空闲，再以同样规则运行并冻结 UR5 LOBO；
6. 只有两项均成功才发布总 `final_receipt.json`。Piper 任一失败都会抛出并终止循环，UR5 不会
   启动。

所有包含 `fresh` 或 `confirmation` 路径分量的输入、代码、输出、日志和锁都会被拒绝。watcher
输出、Piper 输出和 UR5 输出必须是三个预先不存在且彼此不嵌套的路径。`detach` 会在创建后台
进程之前同步创建 watcher 根、`launch_plan.json` 和排他 `launch.lock`；因此 SSH 命令返回时，
两个外部训练输出已经通过不可变 plan 预注册，重复运行会直接失败。

## 固定训练契约

- 顺序：`train_lobo_piper`，然后 `train_lobo_ur5`；
- 每个本体：`source_body_clock` 与 `body_agnostic` 两个变体；
- 每个变体：5 个 ensemble 成员，seed 为
  `20260828..20260832`，每成员 3000 steps；
- checkpoint 只由 source validation 选择；目标 development 只在十个 checkpoint 都选完后打开；
- held-out target train 不打开；所有 ordinary test HDF5 不打开；
- watcher 自身不会导入 `h5py`，只哈希 label-free manifest、冻结 split、训练输出 checkpoint 和
  JSON 收据。

每个子任务都会生成：

```text
<watcher>/stages/train_lobo_piper/{run.log,run.exit,stage_receipt.json}
<watcher>/stages/train_lobo_ur5/{run.log,run.exit,stage_receipt.json}
```

成功的 `run.exit` 内容严格为 `0\n`；非零退出、超时、训练期间出现其他 GPU compute PID、
输入变化或输出审计不完整都会失败关闭。

每个训练 stage 都以 `start_new_session=True` 进入独立进程组，并要求实测 `PGID == PID`。无论
正常退出还是异常退出，watcher 都同时证明直接进程已回收且整个进程组已消失；若存在残留子进程，
先向整组发送 `SIGTERM`，超时后发送 `SIGKILL`。一旦已经尝试 `Popen` 却不能完整证明 PID/PGID
绑定及两级回收，共享 GPU 锁会保留，输出树也不会冻结，避免后续训练误用仍可能存活的进程。

总终态使用隐藏发布协议：`final_receipt.json`（或 `failure_receipt.json`）与 `run.exit` 先以 mode
`000` create-once 写入；其余文件和目录冻结为 `0444/0555` 并完成逐项验证后，先发布
`run.exit`，最后一步才把 terminal receipt 改为 `0444`。冻结或验证失败时按创建 inode 删除隐藏
终态，因此不会遗留可见的成功收据或 `run.exit=0`。

## 远端预检与 detached 启动

watcher 代码必须部署到一个新的只读目录，例如
`/home/user/etsf_multibody_lobo_watcher_code_r8a_20260828`。LOBO 训练器继续使用已经部署且冻结的
`/home/user/etsf_multibody_lobo_code_r8_20260828`，watcher 不向该目录写文件。

先在服务器读取已经存在的 source63 plan 的字节 SHA；不要猜测，也不要在 source63 尚未完成时
读取其数据目录中的任何 HDF5：

```bash
sha256sum /home/user/etsf_smolvla_schema5_native_source_training_r7c_20260828/launch_plan.json
```

把结果填入下方 `SOURCE_PLAN_SHA`。先运行 `preflight`；它只读代码、plan、manifest 字节、event
spec 和两份 split JSON，不打开任何 group/rollout HDF5：

```bash
PY=/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python
WATCHER=/home/user/etsf_multibody_lobo_watcher_code_r8a_20260828/scripts/launch_multibody_lobo_autonomous.py
SOURCE_PLAN_SHA=REPLACE_WITH_THE_64_HEX_SHA256_FROM_THE_PREVIOUS_COMMAND

$PY "$WATCHER" preflight \
  --code-root /home/user/etsf_multibody_lobo_code_r8_20260828 \
  --source-training-root /home/user/etsf_smolvla_schema5_native_source_training_r7c_20260828 \
  --source-launch-plan /home/user/etsf_smolvla_schema5_native_source_training_r7c_20260828/launch_plan.json \
  --source-launch-plan-sha256 "$SOURCE_PLAN_SHA" \
  --stage1-root /home/user/etsf_stage1 \
  --stage1-source-manifest /home/user/etsf_stage1/source_manifest.json \
  --stage1-source-manifest-sha256 5326b676c43f87dbb02c5b45dc9575aec44dc510fc7a309619b197985bda9e1c \
  --stage1-target-manifest /home/user/etsf_stage1/target_rollouts/target_rollout_manifest.csv \
  --stage1-target-manifest-sha256 89293270eb6590f96732e3ff58292fb81098eeeb22ff4ce7991651df9bcfe794 \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --event-spec-sha256 8b1ff070ee7f9519707f45c42209b266a515b45c95d14f13bb4a95f283f2bff5 \
  --openvla-schema5-manifest /home/user/etsf_openvla_event_branches_v7_development250_20260827/manifest.json \
  --openvla-schema5-manifest-sha256 b611d6604ecd323e90dea15c1bfb6bed6fb22bf974cf77b52e8eb02c0409ef23 \
  --piper-split /home/user/etsf_multibody_lobo_split_piper_r8_20260828.json \
  --piper-split-sha256 a21765e97f6d1ad50979540820c863a9829bbfdddb76912a711592c17d6fd13b \
  --ur5-split /home/user/etsf_multibody_lobo_split_ur5_r8_20260828.json \
  --ur5-split-sha256 de853b5884d379b5647dba3583b95d1d7d49dfccd2f234da0325ddf6da4865d3 \
  --output /home/user/etsf_multibody_lobo_autonomous_r8_20260828 \
  --piper-output /home/user/etsf_multibody_lobo_piper_train_r8_20260828 \
  --ur5-output /home/user/etsf_multibody_lobo_ur5_train_r8_20260828 \
  --python-bin "$PY" \
  --gpu-index 0 \
  --gpu-lock /tmp/etsf_smolvla_schema5_source63_gpu0.lock
```

预检成功后，将命令中的 `preflight` 改为 `detach`，其余绑定参数保持逐字相同，并追加：

```bash
  --detach-receipt /home/user/etsf_multibody_lobo_autonomous_r8_20260828.detach_receipt.json \
  --detach-log /home/user/etsf_multibody_lobo_autonomous_r8_20260828.launcher.log
```

不要额外套 `nohup`；`detach` 使用 `start_new_session=True`，后台 watcher 与之后两个训练任务均不
依赖 SSH 客户端。

## 总成功收据契约

总收据是：

```text
/home/user/etsf_multibody_lobo_autonomous_r8_20260828/final_receipt.json
```

强依赖下游必须同时校验：

- `format == "etsf_multibody_lobo_autonomous_watcher_v1"`；
- `status == "complete_sequential_piper_then_ur5_lobo_training"`；
- 删除 `receipt_sha256` 字段后计算 canonical JSON SHA-256，必须等于 `receipt_sha256`；同时在
  下游预注册中记录该文件的字节 SHA-256；
- `execution_order == ["train_lobo_piper", "train_lobo_ur5"]`；
- 两个 `stage_results` 均为 `status=complete`、`returncode=0`，并绑定各自
  `run_exit_sha256`、`log_sha256`；
- 两个 `artifact_audit.status` 都是
  `training_and_frozen_target_development_evaluation_complete`，held-out body 分别是 `piper` 与
  `ur5-wsg`，并提供 `summary_sha256`、十个 checkpoint SHA 和完整输出 inventory SHA；
- 全局及两个 artifact audit 的 `target_unused_train_payload_opened=0`、
  `test_group_hdf5_opened=0`；全局 `watcher_hdf5_opened=0`、
  `test_hdf5_opened_by_watcher=0`、`test_labels_read_by_watcher=false`；
- `artifacts_frozen_read_only=true`，且两个外部训练输出树与 watcher 输出树实际上均无写权限。

这份收据证明严格的 zero-target-label LOBO 预测实验完成，不自动证明机器人任务成功率提升；后者
仍需成对的 baseline-vs-plugin 在线执行实验。

## 本地验证

```bash
python3 -m py_compile scripts/launch_multibody_lobo_autonomous.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_launch_multibody_lobo_autonomous.py
```

定向测试覆盖上游失败门禁、上游只读终态、split/输入 SHA 篡改、连续空闲确认、逐子任务
log/exit、LOBO 十 checkpoint 绑定、test-HDF=0、create-once 预注册与 detached 新 session。

## 2026-08-28 r8d 与 r7g 的当前绑定

旧 r8c 的 static plan 不可变绑定 r7e，因此没有原地修改。确认 r8c 仍为 source receipt 等待态、
`stages_started=[]`、两个外部训练输出均不存在后，只终止了该单一 watcher 进程；旧输出、plan、
detach receipt 和日志全部保留。

当前有效 watcher 使用新物化且只读的代码/输出根：

```text
watcher code /home/user/etsf_multibody_lobo_watcher_code_r8d_20260828
trainer code /home/user/etsf_multibody_lobo_code_r8d_20260828
watcher root /home/user/etsf_multibody_lobo_autonomous_r8d_20260828
Piper output /home/user/etsf_multibody_lobo_piper_train_r8d_20260828
UR5 output /home/user/etsf_multibody_lobo_ur5_train_r8d_20260828
pid 1945067（detached，PPID=1，SID=1945067）
```

r8d 精确绑定 r7g 的 `launch_plan.json` 文件 SHA
`27eb78fea12ce6ea0e2b7e86809f1e64b6a63557c86414895c95809ea158ccdf`，并与 r7g 共用新的锁
`/tmp/etsf_smolvla_schema5_source63_r7g_gpu0.lock`。其 static plan SHA
`917d456242120ef52c3a71ad14fc7840ce93ad5f8189c5121296ba9cf3b7f7b3` 已在 plan、state 和 detach
receipt 间核对一致。当前状态是 `waiting_for_authenticated_source63_terminal_receipt`，没有读取
test HDF、没有启动 Piper/UR5 stage。r7g 完成并释放其精确所有权锁后，r8d 才会按 Piper→UR5
顺序继续。

## r8e 严格 source deployment binding（绑定 r7h，已部署）

r8e 不把 960D SmolVLA-native `policy_feature_action_bridge` 复制进 96D
`MultibodyCanonicalEventWorldModel` checkpoint。两个 LOBO ensemble 只用于跨本体科学评测，必须明确
记录 `lobo_checkpoints_rerank_authorized=false`；在线 rerank 的唯一部署 checkpoint 仍是 r7h 原生
`counterfactual_ensemble.pt`。

r7h 终态 gate 强制要求并验证 `training_audit.ensemble_checkpoint`、checkpoint 文件 SHA 与
`policy_feature_action_bridge_sha256`。source 根必须保持完全只读，ensemble 必须是该根内的普通文件，
且其实际字节 SHA 必须与 final receipt 一致。gate 同时计算并保留 source final receipt 的文件 SHA 与
canonical logical SHA。

取得终态后、申请 GPU 锁前，watcher 在新输出根 create-once 写入
`source_binding_receipt.json`。该收据内容寻址绑定：

- 已预注册 source launch plan 的路径、文件 SHA 和 static logical SHA；
- source final receipt 的路径、文件 SHA、logical SHA 和终态；
- 原生 source ensemble 的路径、文件 SHA、SmolVLA checkpoint family 与 policy bridge contract SHA；
- `deployment_rerank_authority=native_source_ensemble_only`；
- `lobo_checkpoint_role=canonical_cross_embodiment_scientific_evaluation_only`；
- `lobo_checkpoints_rerank_authorized=false`。

LOBO watcher 不二次反序列化 torch checkpoint。这里的 header 信任根是：外部 SHA 固定的 source
launch plan，以及已冻结 source final receipt 中由 source launcher 完成的 safe checkpoint load、
ensemble/member bridge header 一致性验证。r8e 自身重新哈希 final receipt 和 ensemble 字节，并在每个
stage 前、stage 输出验证时和最终收据生成前重验 source binding；因此不需要把 torch、bridge verifier
和 policy adapter 实现引入 LOBO watcher 的独立代码 closure。

Piper、UR5 两个 stage receipt 和 artifact audit 都必须携带同一个内容寻址
`source_binding_contract`、原生 `deployment_rerank_checkpoint` 与 bridge SHA。缺字段、任一 SHA/路径被
篡改、source 树重新变为可写、stage contract 不一致，都会 fail closed。最终 LOBO receipt 再次要求
两条 stage 完整一致后才发布相同绑定；它不会授权任何 LOBO checkpoint 执行 actor override。

这一改动只能部署到新的代码根和新的 r8e watcher/Piper/UR5 输出根，不能原地修改已冻结 r8d。

计划使用的全新远端根为：

```text
watcher code /home/user/etsf_multibody_lobo_watcher_code_r8e_20260828
trainer code /home/user/etsf_multibody_lobo_code_r8e_20260828
watcher root /home/user/etsf_multibody_lobo_autonomous_r8e_20260828
Piper output /home/user/etsf_multibody_lobo_piper_train_r8e_20260828
UR5 output /home/user/etsf_multibody_lobo_ur5_train_r8e_20260828
shared lock /tmp/etsf_smolvla_schema5_source63_r7h_gpu0.lock
```

部署验收结果：

```text
watcher PID 1960033（PPID=1，SID=1960033）
launcher SHA256 3af8933fa5ccd09e7b06dc1912926510e5a9fb0508b2aee3c9d323adafb71206
canonical trainer SHA256 2cb71e5a3bbf0e92d8ce3c493f4a66ab394497cd538cde9ad5db6a3e62a5a32f
LOBO trainer SHA256 25f5b2324996749b4331aab0a720d7c0f944dc573d36ee2375d9861b698fbf08
remote trainer bundle logical SHA256 ed8549abd040525e3d5d75d86ab9845d72c5cedc795c094c3d614cc7bd32e36d
static plan logical SHA256 49976a7b58fbcdebc698ffef09715646f4c35a8326585269fdb81e9f6b92f038
launch_plan.json file SHA256 5523bf5063c85287e5d558fd913fc5da75ca360b5623868768227e0579b3784d
detach receipt logical SHA256 7112fcadacc69b18fefd5a0bcc3f5c082d0048824363971f63328073d4131733
detach receipt file SHA256 aaeb807b90d56b1ddf6bbae1123ef047c1f4ef3f3db8e00010ad145beab7f335
```

watcher、trainer 两个代码根只有三个预注册文件，文件均为 `0444`、目录均为 `0555`，无符号
链接。跨两个 30 秒轮询周期后，进程身份保持不变，状态为
`waiting_for_authenticated_source63_terminal_receipt`，`stages_started=[]`、
`stage_lifecycles=[]`，Piper/UR5 输出均不存在，且 watcher 尚未申请 GPU 锁。它会在 r7h 发布并
通过认证的冻结终态后才继续。

部署前验证为定向 `18 passed`、source/bridge/LOBO 联合 `122 passed`、全仓 `890 passed`；唯一
警告来自本地 CPU-only Torch 的 CUDA 驱动探测，不是 GPU 训练或测试失败。
