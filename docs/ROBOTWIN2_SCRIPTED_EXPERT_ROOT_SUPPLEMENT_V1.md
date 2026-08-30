# RoboTwin2 scripted-expert root 补充集 v1

这个补充集只解决正式 on-policy 分支缺少 e3/e4 临界监督的问题，不替代、修改或追加到正式 8000-branch 数据根。每个场景先由 public RoboTwin `move_can_pot.play_once()` scripted expert 推进；采集器在每次真实 `scene.step()` 后按冻结 analytic event spec 观察状态，只冻结首次 e3 和首次非终局 e4。根冻结以后 scripted planner 立即结束，四候选和所有 continuation 都来自与 primary binding 相同的 frozen actor。

## 冻结设计

- 本体：`aloha-agilex / arx-x5 / franka / piper / ur5`
- 条件：`clean / randomized`
- 根事件：每个 seed 各一个 e3、一个 e4；不取同事件相邻帧
- seed：`2026081000..2026081004`
- seed 到新 actor 分支预算的标签盲映射：
  - `2026081000 → H=10`
  - `2026081001 → H=25`
  - `2026081002 → H=50`
  - `2026081003 → H=100`
  - `2026081004 → H=200`
- 总规模：100 decisions / 400 branches

H 从 expert root 重新计 actor action：`take_action_cnt=0`、`eval_success=False`、`plan_success=True`、`step_lim=H`。H 不由 expert 已执行帧数、候选动作或任何 terminal outcome 决定。快照保留物理状态、仿真时钟、RNG 和 expert 前缀；fresh scene restore 后严格核对冻结的 object registry，再只执行一次 contact-cache canonicalization step。

若固定 seed 没有出现某个目标事件，manifest 只记录 missing target，不允许改取相邻帧、换 seed 或根据候选 outcome 搜根。五本体 binding materializer 会因设计不完整而拒绝产出正式 binding。

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

必须在远端 RTX 4090 和 public RoboTwin checkout 上运行。采集器不读取 official expert ZIP；scripted expert 在线使用五个预注册 fresh seeds。输出包括：

- `manifest.json`：独立、签名的 `proper_world_supplement_manifest_v1`，含 actor/checkpoint/code/root selection provenance；
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

materializer 调用 trainer 的 primary binding validator，逐本体绑定 authority 中的 checkpoint tree SHA，验证完整 5×2×2×5 设计，并拒绝任何 `(body, condition, seed)` 与 primary 重叠。它只读 manifest 元数据，不打开 transition NPZ。输出是 create-once：已有 binding 只有在字节完全一致时才接受，任何不一致 final 或 `.partial` 都拒绝覆盖。

trainer 对这个 binding 的用途固定为 source-body train-only proper world losses；不进入 normalization、rank/utility、source validation、checkpoint selection 或 calibration，outer-heldout supplement manifest 与 payload 保持 zero-open。

## 本地无仿真验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_collect_robotwin2_scripted_expert_root_actor_branches_v1.py \
  tests/test_collect_robotwin2_five_body_ee_candidate_branches_v1.py
```

测试使用 fake scene 和合成 manifest，不启动 SAPIEN/RoboTwin，也不占用 GPU。
