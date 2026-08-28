# Formal Frozen-Core 目标适配训练契约

## 当前结论

`event_world_model_best.pt` 的权威审计结果是
`state_input_dim=4096`、`action_dim=14`、`num_bodies=1`、`num_policies=1`；body
注册表只有 `piper_piper_0.6:0`，policy 注册表只有 `OpenVLA:0`，并且没有双目标预留或
source-only 重训练证明。因此它**不能直接进行 SmolVLA 目标适配**。正式训练器会在读取目标
结构化数组之前失败关闭，不会把事后扩展的随机 embedding 当作可迁移共享核心。

解锁训练需要先完成一次与目标数据隔离的 source-only 流程：

1. 在不知道目标样本和标签的条件下，同时为 target body 和 `smolvla` policy 预留行；
2. 使用原始、内容绑定的 source manifest 和 exact source split 重训 shared core；
3. source batch 不得引用两条预留行，且两行在 source-only 重训中必须 bit-exact 不变；
4. 将 source manifest/split SHA、重训步数/组数、两行 SHA 和完整 source state SHA 写入并签名为
   `etsf_formal_dual_target_reservation_v1`。

单轴 `transfer_source_core_expansion` 或旧式 `reserved_source_retraining` 不能替代这个证明。

## 训练边界

入口是 `scripts/train_formal_frozen_core_transfer_adapters.py`。它只训练：

- `StateAdapter(960→4096)`；
- policy action adapter 和 body action-effect adapter；
- decision-step clock 的 `beta` 与 step scale；
- source-only 阶段预留的 target body/policy 两行。

其余 shared-core 张量在每个优化步后恢复，并按 bit-exact digest 审计。这里的时间目标仅称为
`decision-step duration`，不能解释成秒或固定 250 Hz 时长。

输入格式为 `etsf_formal_transfer_structured_arrays_v1`，必须绑定 event spec SHA、schema6 object
registry SHA 和 pose-integrity spec SHA。监督包括 next event、destination、decision-step duration、
post-predicate 和物理对象变化。对象损失只使用
`object_delta_supervision_valid[:, object_feature_object_index]`；该布尔值必须逐对象逐步严格等于
`invalid_reason_bitset == 0`，所以 reset、teleport、时间戳异常或物理发散区间不会进入对象头监督。

success/recovery 标签必须在 logical group 内一致。每个二分类头只有在正、负类各至少 50 个独立
logical group 时才启用；recovery 还要求 source core 明确带有已监督 recovery head。

## 输出与禁止事项

成功运行只生成内容寻址、只读的 `.pt` checkpoint 和 JSON receipt。两者都声明：

- `monitor_only=true`；
- 不授权 selection、action ranking、环境执行或跨本体结论；
- 不授权 shared-core 梯度或更新；
- 未读取 Fresh/confirmation 数据。

该模块没有模拟器、采集器、actor 或 selector 接口。它的产物只能进入后续的冻结、配对、同 seed、
预注册非 Fresh 评估；在独立目标本体与目标 policy 上通过预注册门槛前，不得声称跨本体有效。

## 本地合成验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  tests/test_train_formal_frozen_core_transfer_adapters.py
```

测试覆盖 CPU 训练、双预留签名、现有单行 checkpoint 的读取前失败关闭、schema6/tensor 篡改、
独立组 support gate、immutable-core 恢复审计、内容寻址 checkpoint 和 receipt 自签名。
