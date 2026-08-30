# RoboTwin2 scripted-expert root 补充集 v1

这个补充集只解决正式 on-policy 分支缺少 e3/e4 临界监督的问题，不替代、修改或追加到正式 8000-branch 数据根。每个场景先由 public RoboTwin `move_can_pot.play_once()` scripted expert 推进；采集器在每次真实 `scene.step()` 后按冻结 analytic event spec 观察状态，只冻结首次 e3 和首次非终局 e4。根冻结以后 scripted planner 立即结束，四候选和所有 continuation 都来自与 primary binding 相同的 frozen actor。

## 冻结设计

- 本体：`aloha-agilex / arx-x5 / franka / piper / ur5`
- 条件：`clean / randomized`
- 根事件：每个被选中的 reset seed 各一个 e3、一个 e4；不取同事件相邻帧
- 每个 `(body, condition, horizon_slot)` 在任何 rollout 前冻结 16 个有序 reserve seeds；五个 slot 固定对应 `H={10,25,50,100,200}`
- 全部 roster 使用互不重叠的 `2026081000..2026081799`，按 body、condition、slot、reserve index 顺序确定；正式 primary 从 `2026082000` 开始
- 每个 slot 只接受有完整 e3/e4 pair、且两个 root 都通过 fresh-restore canonicalization 的第一个 seed
- 不同本体独立解析自己的 roster，绝不取“五本体共同成功 seed”，因此 heldout 本体的可用性不会改变 source 本体数据
- 总规模：100 decisions / 400 branches

H 从 expert root 重新计 actor action：`take_action_cnt=0`、`eval_success=False`、`plan_success=True`、`step_lim=H`。H 不由 expert 已执行帧数、候选动作或任何 terminal outcome 决定。快照保留物理状态、仿真时钟、RNG 和 expert 前缀；fresh scene restore 后严格核对冻结的 object registry，再只执行一次 contact-cache canonicalization step。

root capture 与 canonicalization 是独立的预选择阶段。在一个 seed 的 e3/e4 均确认以前，不生成或执行任何 actor candidate outcome。每次 fresh setup 前都用真实 requested seed 重置 Python RNG（RoboTwin 本身只重置 NumPy/Torch，而 CuRobo fallback 会使用 Python `random`），因此重启时跳过已拒绝 attempt 不会改变后续 slot 的规划随机性。`UnStable`、expert plan failure、missing e3/e4 和 canonicalization failure 都以 `rejected_before_actor_outcomes` 写入 attempt ledger，再严格推进到同一 slot 的下一个预注册 seed；不能改取相邻帧、把 eK 当 e4、放宽 event 阈值或依据候选 success/stage 搜 seed。16 个 seed 全部耗尽时 collector 写 `reserve_exhausted` 并非零退出，不得标记完成。

每个被选中的完整 root pair 在第一条 actor outcome 前原子保存到 `root_pairs/*.pt`，manifest 绑定文件 SHA。进程中断后直接从该 pair 恢复，不重新依赖 public expert/CuRobo 产生逐位相同的前缀。已有 group 只有在 group NPZ 与 diagnostics NPZ 都存在且 SHA 完全匹配时才允许跳过；缺失或篡改一律 fail closed。只有精确 10 个 selected slots、20 groups、完整有序 reject history 和全部文件 SHA 通过时，`collection_status` 才能写为 `complete`。

## 独立输出与采集

每个本体使用独立目录，例如 `$SUPPLEMENT_ROOT/piper`。collector 需要 formal primary 使用的同一个 actor checkpoint 和 actor authority：

```bash
python scripts/collect_robotwin2_scripted_expert_root_actor_branches_v1.py \
  --body piper \
  --actor-checkpoint /path/to/pretrained_model \
  --actor-authority /path/to/frozen_actor_authority.json \
  --vlm-metadata-path /path/to/smolvlm_metadata \
  --robotwin-root /path/to/RoboTwin \
  --event-spec /path/to/analytic_event_spec.json \
  --output "$SUPPLEMENT_ROOT/piper"
```

必须在远端 RTX 4090 和 public RoboTwin checkout 上运行。采集器不读取 official expert ZIP；scripted expert 在线按本体局部的冻结 reserve roster 使用 fresh seeds。输出包括：

- `manifest.json`：独立、签名的 `proper_world_supplement_manifest_v1`，含 actor/checkpoint/code/root selection provenance；
- `root_pairs/*.pt`：被选 root pair 的 create-once 恢复文件；
- `groups/*.npz`：与 primary canonical group 相同的训练数组；
- `groups/*.diagnostics.npz`：非训练候选差异与执行诊断。

## 五本体 binding

五个本体均达到 20 decisions 后，在它们的共同父目录创建一次 binding：

```bash
python scripts/materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py \
  --primary-binding /path/to/primary-training-binding.json \
  --actor-authority /path/to/frozen_actor_authority.json \
  --body-manifest "aloha-agilex=$SUPPLEMENT_ROOT/aloha-agilex/manifest.json" \
  --body-manifest "arx-x5=$SUPPLEMENT_ROOT/arx-x5/manifest.json" \
  --body-manifest "franka=$SUPPLEMENT_ROOT/franka/manifest.json" \
  --body-manifest "piper=$SUPPLEMENT_ROOT/piper/manifest.json" \
  --body-manifest "ur5=$SUPPLEMENT_ROOT/ur5/manifest.json" \
  --output "$SUPPLEMENT_ROOT/binding.json"
```

materializer 调用 trainer 的 primary binding validator，逐本体绑定 authority 中的 checkpoint tree SHA，验证完整 5×2×2×5 selected-slot 设计、每个 selected seed 前的有序 reject ledger，并拒绝任何真实 `(body, condition, requested_seed)` 与 primary 重叠。它只读 manifest 元数据，不打开 transition NPZ；binding 额外记录 50 个 selected seeds、reject 总数以及每本体 roster/selection SHA。输出是 create-once：已有 binding 只有在字节完全一致时才接受，任何不一致 final 或 `.partial` 都拒绝覆盖。

trainer 对这个 binding 的用途固定为 source-body train-only proper world losses；不进入 normalization、rank/utility、source validation、checkpoint selection 或 calibration，outer-heldout supplement manifest 与 payload 保持 zero-open。

## 本地无仿真验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_collect_robotwin2_scripted_expert_root_actor_branches_v1.py \
  tests/test_watch_robotwin2_postformal_shared_head_upgrade_v1.py
```

测试使用 fake scene 和合成 manifest，不启动 SAPIEN/RoboTwin，也不占用 GPU。
