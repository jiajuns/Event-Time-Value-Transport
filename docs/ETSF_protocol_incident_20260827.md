# ETSF test-seed 协议事件（2026-08-27）

## 事件

schema-v5 collector 的首次真实 smoke 使用了默认 `--limit 1`，解析为 RoboTwin requested /
resolved seed `100100000`。事后与既有 split manifest 交叉检查确认，该 seed 位于原 25-seed
`test` split。smoke 产物为：

```text
/home/user/etsf_openvla_event_branches_v5_smoke1_20260827
schema_version: 5
seed: 100100000
candidate success: [0, 1, 1, 0]
```

该数据只用于验证真实 HDF5 schema、query hidden 链和 terminal 边界，没有参与模型训练、
checkpoint 选择、温度校准或 guard 选择。但是候选终局标签已经出现在采集日志和 schema
审计输出中，所以原 test split 不再满足“全部标签从未查看”的严格定义。

schema-v3/v4 smoke 使用 seed `100100001`，属于 train split，不受此事件影响。正在运行的
正式 schema-v5 train100 明确使用 `--seeds-key train`，不包含 test seed。

## 处置

1. 不删除或改写 smoke 数据和日志，保留完整审计证据。
2. 原 25-seed test 后续若评测，只能标为 development holdout，不能称为 untouched sealed
   confirmation；也不能静默删除已查看的 seed 后把剩余 24 个重新命名为原 sealed test。
3. 反事实训练器在 split 前只读取 identity attrs，train/validation 之外的 HDF5 标签不加载。
4. 正式成功率结论需要新建、预注册且在模型/guard 冻结前不执行的 fresh confirmation
   seed 集。若 RoboTwin 固定 seed 池无法扩展，则必须把结论降级为 development evidence。
5. collector 后续 smoke 必须显式提供 `--seeds-file ... --seeds-key train` 或 `--seeds`，不再
   使用默认 `--limit/--offset` 选择。

## 追加：retry2 策略别名启动失败

排序损失修复后的首次 `retry2_rank` 启动在读取第一个 schema-v5 group 时退出，未执行任何
训练 step。factual checkpoint 用完整 OpenVLA-OFT checkpoint 路径作为 `policy_to_id` 键，
而 collector manifest 将同一策略记录为 `openvla`；新增的跨策略 fail-closed 校验把两者
误判成不同策略。失败输出与日志保留在：

```text
/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry2_rank_20260827
/home/user/etsf_openvla_counterfactual_v5_move_can_pot_retry2_rank_20260827.launch.log
```

处置是加入内容无关的策略身份规范化：名称或 checkpoint 路径只要明确包含 OpenVLA / SmolVLA，
分别映射为 `openvla` / `smolvla`；若两个别名映射到同一规范名但 ID 不同则拒绝。远端预检确认
100 个 descriptor 均为 schema 5，规范化后的 checkpoint/collector 策略均为 `openvla:0`。
随后仅在全新 `retry2b_rank` 目录重启，未覆盖失败记录。

## 追加：开发集采集中途日志意外暴露

2026-08-27 15:23 CST，为核对远端采集进度执行了 `tail`，输出不仅包含运行状态，还包含了
开发扩展集 group 93--108 的候选终局标签。该暴露发生在动作敏感共享头、OOF 划分、固定训练
预算、四候选部署守门规则、结构化预测诊断与最终 artifact 绑定均已实现，并在本地完整测试
`154 passed` 之后；随后远端只同步了这批已通过测试的同一文件，针对性测试为 `19 passed`。

处置如下：

1. 不依据已看到的中间标签修改模型、损失、超参数、候选集合、OOF 划分或授权阈值；这些选择
   从该时刻起视为冻结。
2. 第五候选 `sample_blend_1.000` 仍严格限定为训练增强，成功率授权与 fresh confirmation 只使用
   预注册的前四个部署候选。
3. 后续进度检查只读取 `status`、`completed`、进程状态和文件计数，不再输出 `groups`、
   `success` 或其他标签字段。
4. 允许的后续变更仅限与标签无关且有回归测试覆盖的运行故障修复；任何此类变更都必须另行记录，
   不能用该开发集进行自适应调参。
5. 最终提升结论仍只由尚未开启、守门通过后才允许访问的 fresh50 确认集决定。
