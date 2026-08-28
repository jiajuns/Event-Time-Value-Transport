# SmolVLA 共享状态与 schema-v5 采集契约

> 只读审计日期：2026-08-27。本轮只读取了 4090 主机的源码、模型配置和既有 HDF5，
> 没有启动 GPU 推理、训练或 RoboTwin 采集。

## 结论

SmolVLA 可用于事件世界模型的候选共享状态是 **flow-matching denoise 之前的 VLM prefix
输出**，不是既有文件中的 720 维 `candidate_hidden`。当前审计版本的具体锚点为：

```text
images + fixed language + robot state
        ↓ embed_prefix
SmolVLM prefix transformer
        ↓ text_model.norm
最后一个未 padding 的 state token，960-D       ← 保存为 query_hidden
        ↓ fill prefix KV cache
explicit candidate noise + timestep
        ↓ 10-step action-expert denoise
720-D expert hidden / 50×14 action chunk          ← expert hidden 禁止作为共享状态
```

远端实际加载的是 LeRobot `0.4.4`：

- 源码：`/home/user/etsf_stage0/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py`
- 源码 SHA256：`3bdbaeecbd0dd3908d08507c13ed3517e63d2a653555322e2428066efb77b5f4`
- VLM/expert bridge：`smolvlm_with_expert.py`，SHA256
  `b70356145870c7da1e92a2195ee4626f7e9f9387576c6bed5ba2bfefae2a38d9`；
- `sample_actions()` 先调用 `embed_prefix()`，再运行 VLM prefix forward 填 KV cache，最后才进入
  `denoise_step()` 循环；
- metadata 的 VLM `text_config.hidden_size=960`，SmolVLA checkpoint
  `prefix_length=0`、`chunk_size=50`、`max_action_dim=32`，实际 ALOHA action 为 14 维。

三个现有 fixed-language split 均为 schema-v2，只有 terminal success/cost：

| split | groups | 已有 hidden | 实测候选最大差异 | hook 次数 | dense trajectory |
|---|---:|---|---:|---:|---|
| train | 28 | `4×720` expert hidden | `0.2306` | 每候选 10 | 无 |
| validation | 10 | `4×720` expert hidden | `0.3165` | 每候选 10 | 无 |
| sealed test | 10 | `4×720` expert hidden | `0.4071` | 每候选 10 | 无 |

这些数值直接证明 720-D 表征随 flow noise 改变。它仍可用于既有 direct-Q/pairwise baseline，
但不能输入声称“候选共享 observation state”的事件模型。旧 collector 已新增显式
`candidate_hidden_contract=action_expert_noise_conditioned_not_shared_observation_state`。

## 已实现 collector

`scripts/collect_smolvla_etsf_event_branches.py` 生成可被现有 structured counterfactual loader
直接读取的 schema-v5 HDF5：

- 根级 `initial_hidden/pre_hidden/post_chunk_hidden` 使用 960-D VLM prefix state；
- HDF 根属性冻结 `event_spec_sha256`、SmolVLA modeling/bridge 源码 SHA256，并将
  hook anchor、960-D、`prefix_length=0`、noise boundary 与两个源码 hash 编入
  `shared_state_contract_id`；resume、trainer 与在线插件均逐层核对；
- 对同一 query 的每个 explicit-noise candidate 独立捕获 prefix，要求 bit-exact 相等；
- `candidate_hidden` 字段被 validator 明确禁止；
- 候选 0 名为 `deterministic`，含义是由 `(scene_seed, query_index, candidate_index=0)` 固定的
  actor fallback，不是零噪声；
- 每个 candidate 分支从相同 requested/resolved seed、相同 instruction 和相同 simulator
  pre-state 重置；
- 保存逐 step `object_poses/proprio`，以及每个 query 的 action、连续 executed-prefix mask、
  post state 和 query 边界；
- 保存 canonical event/time 标签；训练 loader 会再次从对象轨迹推导动态 predicates、
  regression/recovery 和 object delta，而不盲信缓存标签；
- terminal query 仍做一次只读 policy forward 以获取 future state target，但其动作不执行。

manifest 将 SmolVLA policy 与 VLM/expert bridge 两个源文件 SHA、event-spec SHA、state anchor、fixed-language contract、
requested/resolved seeds、动作 chunk 和每 query 的共享状态相等性一起冻结。resume 时任一字段改变
都会拒绝复用旧目录。

## fail-closed 边界

当前 hook 只支持已审计布局，并主动拒绝：

- `prefix_length != 0`，因为最后一个 token 可能是 padding；
- 找不到 `get_vlm_model().text_model.norm`；
- prefix hook 每次 query 不是恰好调用一次；
- 任意候选的 960-D prefix state 与候选 0 不逐位相等；
- 所有 flow-noise 候选动作与 deterministic 候选完全相同，或 noise seed 重复；
- hidden 维度不是 960、chunk 不是 50、actor action 不是 14-D；
- 分支 instruction、resolved seed、对象顺序或 simulator pre-state 不一致；
- query hidden chain、action mask、轨迹边界或 dense event 标签不一致。

因此本地 CPU 测试通过只说明 schema/validator/训练 loader 接口成立，不代表真实 SmolVLA hook
已经在 4090 上通过，也不代表成功率提高或跨本体迁移成立。

## 验证命令

已完成的 CPU 测试：

```bash
python3 scripts/collect_smolvla_etsf_event_branches.py --self-test
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_collect_smolvla_etsf_event_branches.py
```

部署代码后应先执行 4090 **接口 smoke**，只验证多候选动作不同且 960-D state 完全相同：

在线 rerank 时，`SmolVLAStateAdapter.calibration_id` 必须等于 checkpoint
`contract.state_contracts.smolvla.calibration_id`。这个相等关系只证明采集/训练/部署
使用同一状态定义，不证明跨策略或跨本体有效；后者仍需独立 held-out 验证。

```bash
/home/user/etsf_stage0/.venv_smolvla_robotwin_eval_np126/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/run_smolvla_native_candidates_smoke.py \
  --model-path /home/user/etsf_smolvla_models/smolvla_robotwin_967623a0 \
  --vlm-metadata-path /home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata \
  --output /home/user/etsf_smolvla_shared_prefix_smoke_20260827.json \
  --candidate-count 4
```

smoke 通过后再用一个开发 seed 做 schema-v5 端到端采集；该结果只能标记
`interface/development smoke`，不能进入 sealed success 结论。正式训练/validation/test 应重新
冻结互斥 seed manifest，并确保 sealed 数据在 temperature、scoring 和 guard 冻结前不可读。
