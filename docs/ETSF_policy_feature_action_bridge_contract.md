# ETSF policy feature/action bridge 强契约

“event head 可插拔”不等于 policy latent 可互换。SmolVLA 的共享 query state 是 flow-noise
之前的 960-D contextualized VLM prefix token；OpenVLA 路径使用 policy query 时
`last_hidden_states[:, -action_dim*num_action_chunks-1]` 的 4096-D 表示。两者只共享
`e0/e12/e3/e4/eK` canonical event interface，以及带 feature-valid mask 的 canonical
action-effect chunk interface，不共享 latent 坐标系。

`scripts/etsf_policy_feature_action_bridge.py` 定义
`etsf_policy_feature_action_bridge_v1`。每个 checkpoint 必须在
`contract.policy_feature_action_bridge` 中精确绑定：

- policy 与 checkpoint family；
- state hook 的语义 source、source SHA、维度、state adapter 名称及实现 SHA；
- native/model action 维度、每个 native feature 的 model slot、action adapter 名称及实现 SHA；
- checkpoint 的 policy row，并与 `contract.policy_to_id` 一致；
- 共同的 canonical event/action-effect interface；
- 对以上全部内容计算的 `contract_sha256`。

adapter SHA 不是调用者可自报的实验字符串。builder/verifier 会对当前部署树中的真实实现文件
重新计算组合 SHA：OpenVLA 绑定其 example adapter 与 shared plugin/base 文件，SmolVLA 再
额外绑定自身文件和所继承的 OpenVLA action-adapter 文件；文件缺失、symlink、名称跨 policy
或 SHA 不同都会失败。

部署端另行生成 `etsf_policy_feature_action_runtime_binding_v1`，记录实际加载的 hook 和 adapter。
verifier 要求 checkpoint contract、runtime binding、模型 config 四者逐字段一致。仅维度一致不够，
仅把 adapter 标为 calibrated 也不够；state source SHA、action slots、adapter SHA 或 policy row
任一不同都会 fail closed。特别地，expected policy 为 OpenVLA 时，SmolVLA-native 960-D
checkpoint 会得到明确拒绝，禁止 padding、线性投影或改 policy 名称后直接复用。

完整的 `canonical_event_id_and_reversible_predicates_v1` 只允许
`structured_events=true`，并要求精确的五个 predicate 与四个 relative-transition 名称。
非 structured checkpoint 必须生成较弱的 `canonical_event_id_only_v1`，不得仅凭相同 event ID
声称拥有 reversible predicate interface。两者仍共享经过 feature-valid mask 的 action-effect
接口，mask 会实际进入 action encoder。

两个高层入口已接入 verifier：

- `example_openvla_event_critic_plugin.rerank_openvla_candidates` 必须提供
  `runtime_bridge_binding`；它从已验证 contract 读取 policy row 和 action slots。
- `example_smolvla_event_critic_adapter.adapt_smolvla_query` 必须提供 checkpoint config、
  checkpoint contract 和 `runtime_bridge_binding`；它不会再由调用者只传一个维度就自称兼容。
- 严格在线 SmolVLA 使用一体化 `rerank_smolvla_candidates(plugin=..., ...)`；该函数让同一个
  plugin 重验 runtime receipt 后立即构造输入并 rerank，避免“验证 A、执行 B”的 TOCTOU。

老 checkpoint 未携带这一 contract 时，底层 plugin 仍可用于历史 monitor/离线诊断，但严格的
`EventCriticPlugin.select/rerank` 也会无条件退回 actor fallback；该授权门不受可配置 guard
开关影响。验证成功后，state 与 candidate 分别携带 state/action binding SHA 和同一个
verification SHA，plugin 在 actor override 前再次与自身 receipt 核对。不能在不知道原始
hook/adapter 文件 SHA 的情况下给
老 checkpoint 补写 contract；正确迁移方式是重新从训练/采集 provenance 生成并冻结它。

离线 verifier：

```bash
python3 scripts/etsf_policy_feature_action_bridge.py \
  --checkpoint /ABS/member.pt \
  --runtime-binding /ABS/runtime_policy_bridge.json \
  --expected-policy openvla
```

合成测试不加载 actor、仿真器或封存数据：

```bash
pytest -q tests/test_etsf_policy_feature_action_bridge.py \
  tests/test_openvla_etsf_event_critic_plugin.py
```
