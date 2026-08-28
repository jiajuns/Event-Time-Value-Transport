# Source dense260 v1：reset-identity 预注册冻结器

`scripts/preregister_smolvla_schema5_source_dense260_v1.py` 是一个纯标准库、只读输入、create-once 输出的冻结器。它不导入模拟器或策略，不执行 reset、action、rollout、采集或训练，也不读取轨迹、HDF、reward、成功率、事件或其他监督标签。输出状态明确为 `collection_not_authorized`；它只是后续人工审查和独立采集适配器所需的不可变合同，不是采集启动器。

## 冻结合同

- namespace：`schema5_aloha_source_dense260_20260829_v1`
- 候选 requested seed：从 `2026083500` 开始、步长 1、严格 400 个
- 选择：按候选顺序取前 260 个可用且 `resolved_seed` 唯一的 reset；若这 260 个的 reset identity 不唯一则直接失败，不使用后面的候选替换
- split：对所选 `(namespace, requested, resolved, reset_identity_sha256)` 做冻结的 canonical SHA-256 排序，再切为 train/calibration/validation = 100/80/80
- 隔离：所选三轴在 split 内唯一、split 间分别零重叠；候选池三轴还必须分别与 official150、Source63、既有 development、Piper development300、Formal、evaluation400 零重叠
- 采集合同：8 个动作候选、`action_exec_steps=5`、`max_steps=200`、action chunk 50、action dim 14、schema 5、事件词表 `e0/e12/e3/e4/eK`
- 监督覆盖合同：exec5 事件状态、下一事件和右删失持续时间、成功/失败/恢复、逐步对象 pose/proprio 与对象状态 delta，以及由 8-candidate dispersion 配合独立 calibration split 支持的不确定性校准；这些都是未来采集字段要求，不参与 seed 选择
- 粗上界：260×8 = 2,080 条候选分支，候选分支最多 416,000 simulator steps；若另行执行每组一条 200-step deterministic baseline，则总上界约 468,000 steps。预注册器本身执行 0 step。

`exec5` 是为减少旧 `exec50` 对短暂 e3/e4 的漏采而冻结的开发采样频率。它不意味着已经证明任务成功率或跨本体性能提升；这些结论只能由未参与开发的封存评估给出。

## 输入边界

冻结器需要：

1. 一份完整的 reset-only 候选 manifest。它必须包含全部 400 行、每个 requested seed 最多一次 reset 的 resolved seed 和不披露原始状态的 `reset_identity_sha256`；失败行只能标记 unavailable，不能重试或伪造 identity。reset identity v1 固定为 `(task, instruction_semantics_sha256, initial_scene_state_sha256)` 的 canonical SHA-256，其中 scene state 是按 semantic object name 排序的 canonical float32 world pose；requested/resolved seed、策略、本体、关节和 drive state 均排除。因此它能发现同一语义场景跨 seed 或跨本体重复，又不会因本体关节不同而天然变成“不重叠”。
2. 六份 aggregate-only identity attestation。每份只包含 reference/候选池三轴集合 commitment、三轴 intersection count=0 和自身 canonical SHA；禁止包含 reference identity 原值。

所有输入先验证精确字段、固定 scope、canonical 内容承诺、能力声明和 attestation 完整性，再进行选择或 split。这里的 SHA 字段只证明内容自洽和防止静默改写，不是密码学身份签名，不能独立证明 attestation 的签发者；未来生成器或 collector 获得采集权限前，还必须绑定经审查的签发者身份、不可变来源或外部 trust root。路径中含 `protected`、`fresh`、`confirmation`、`formal`、`evaluation`、`target`、`trajectory`、`label`、`outcome` 或 `hdf` 的文件会在打开前被拒绝；敏感数据集的角色只能写在 aggregate attestation 的 payload 中，不能通过路径或原始 identity 披露。

现有 `preregister_robotwin_development_expansion_seeds.py` 没有 reset-state identity、三轴 aggregate attestation 和严格的 pre-action 路径边界，并且它会导入模拟器执行 reset，因此不能作为本冻结器的安全依赖。本实现只复用了其“按候选顺序选择 resolved-unique seed”的语义，未导入或修改旧脚本。

## 使用方式与当前缺口

在中性、不可变的输入目录中准备一份 reset manifest 和六份 attestation 后，可执行：

```bash
python3 scripts/preregister_smolvla_schema5_source_dense260_v1.py \
  --reset-manifest /immutable/dense260_inputs/reset_manifest.json \
  --identity-attestation /immutable/dense260_inputs/att_00.json \
  --identity-attestation /immutable/dense260_inputs/att_01.json \
  --identity-attestation /immutable/dense260_inputs/att_02.json \
  --identity-attestation /immutable/dense260_inputs/att_03.json \
  --identity-attestation /immutable/dense260_inputs/att_04.json \
  --identity-attestation /immutable/dense260_inputs/att_05.json \
  --output /immutable/dense260_outputs/source_dense260_v1.json
```

输出以 `O_EXCL` 创建并设为只读，包含 `preregistration_sha256` canonical 内容地址；已有输出不会覆盖。

当前真实 reset-only manifest 和六份 attestation 尚未生成，所以还不能据此启动采集。最小下一步是在 4090 服务器上先运行一个单独审查的 reset-only 生成器，固定上述 namespace，并在任何 policy/action/标签访问前产出这七份输入；随后先离线运行冻结器和验证器，再单独评审 collector adapter。旧 Source63 可以保留为已封存的辅助训练数据，但只有通过 Source63 attestation 后才能与 dense260 的 train 部分组合，不能并入新 calibration/validation；本次没有改动任何旧 collector 或 launcher。
