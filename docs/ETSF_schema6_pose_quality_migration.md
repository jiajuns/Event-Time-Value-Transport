# ETSF schema6 物体姿态质量契约与迁移

状态：实现完成、仅通过合成/篡改测试；尚未采集 schema6 数据，也未授权开启 object head。

## 为什么不能继续直接使用 schema5 的 object delta

schema5 保存了 `object_poses`，但没有记录每个姿态是否来自 reset、teleport、时间轴断裂、
simulator 报错或物理发散。有限数值和 HDF 形状正确并不能证明该 delta 是有效监督。
因此 schema6 采用 fail-closed 规则：没有 schema6 质量组的旧数据，其 object 标签仍视为
不可用；不得离线猜测 `pose_quality_valid=True`，也不得原地修改已签名 schema5 文件。

实现位于 `scripts/etsf_schema6_pose_quality.py`，没有 simulator、actor、GPU、Fresh 或网络依赖。

## 规范

一个新 schema6 HDF 根必须先设置：

```python
hdf.attrs["schema_version"] = 6
```

每条候选轨迹继续保存 `object_poses[T,O,7]`，布局固定为：

- 平移：索引 `0:3`，单位 `metre`；
- 四元数：索引 `3:7`，顺序 `wxyz`，旋转单位 `radian`；
- 坐标系：`simulator_world`。

在同一 trajectory group 下新增 `pose_quality_v6/`：

```text
object_poses                         [T,O,7]
pose_quality_v6/
  pose_quality_valid                bool[T,O]
  pose_quality_reason_bitset        uint32[T,O]
  simulator_timestamp_s             float64[T]
  control_step                      uint64[T]
  physics_substep_count             uint32[T]
  reset_generation                  uint32[T]
  reset_flag                        bool[T]
  teleport_flag                     bool[T,O]
  simulator_pose_error_flag         bool[T,O]
  object_registry_json              scalar UTF-8 JSON
  pose_integrity_spec_json          scalar UTF-8 JSON
```

组属性冻结 `frame`、`translation_unit`、`rotation_unit`、`quaternion_order`、
`object_registry_sha256`、`pose_integrity_spec_sha256` 与 `logical_payload_sha256`。
逻辑 payload SHA 同时绑定 poses、所有质量/时间字段、registry 和 spec；读取时重新推导质量位图并
核对 SHA，单独修改标签、pose 或元数据都会拒绝。

registry 的对象顺序就是 `O` 轴顺序。每个对象必须明确提供：

- `name`；
- `stable_sim_actor_id`，不能用显示名称或数组下标代替；
- `asset_model_id`；
- `role`；
- `is_static`。

integrity spec 必须在采集前冻结，明确 registry SHA、frame/quat/unit、时间语义、工作空间、
四元数容差、动态/静态对象单步运动上限、合法 simulator 时间步范围和 physics substep 上限。
`thresholds_fit_from_pose_data` 必须为 `false`，防止先看数据再调阈值。

reason bit 由 `PoseQualityReason` 固定：非有限 pose、四元数范数、越界、单步平移/旋转、
静态物体运动、reset discontinuity、teleport、时间戳错误、control step 错误、physics substep
错误、reset 标志不一致和 simulator 主动报告无效。未知 bit 会在读取时失败。

## 新 collector 的最小接入方式

不要修改或覆盖 schema5 文件。复制 collector 为新的 schema6 collector，并只向新的临时 HDF
写入。对每个 candidate branch：先写 `object_poses`，再调用：

```python
from etsf_schema6_pose_quality import write_pose_quality_v6

receipt = write_pose_quality_v6(
    branch,
    registry=frozen_registry,
    spec=frozen_integrity_spec,
    simulator_timestamp_s=timestamps,
    control_step=control_steps,
    physics_substep_count=physics_substeps,
    reset_generation=reset_generations,
    reset_flag=reset_flags,
    teleport_flag=teleport_flags,
    simulator_pose_error_flag=simulator_pose_error_flags,
)
```

collector 必须从 simulator 的真实 reset/teleport/control loop 记录这些字段，不能事后根据 pose
轨迹猜测。每条 candidate 独立 reset 时，其第一帧要求 `reset_flag=True`、
`control_step=0`、`physics_substep_count=0`；正常帧 control step 必须逐一递增。文件原子发布前，
对所有 branch 调用 `validate_pose_quality_v6`，并把 registry/spec SHA 写入 collection manifest。

## object head 的训练读取

禁止直接计算 `poses[end]-poses[start]` 后全量训练。使用：

```python
from etsf_schema6_pose_quality import load_object_delta_supervision_v6

targets = load_object_delta_supervision_v6(
    branch,
    start_steps=query_steps,
    end_steps=query_post_steps,
    expected_registry_sha256=frozen_registry_sha,
    expected_spec_sha256=frozen_spec_sha,
)
loss_mask = targets["object_delta_supervision_valid"]
delta_xyz_m = targets["object_delta_xyz_m"]
```

区间监督只有在 `(start,end]` 的每个 destination step 都有效，且没有跨 reset/teleport 时才
为真；返回的 interval reason bitset 用于审计被屏蔽原因。训练器还必须按逻辑 trajectory/group
划分 train/validation，不能让相邻帧跨折泄漏。

## 从 schema5 到 schema6 的边界

1. schema5 保持只读、SHA 不变；不能补写质量组。
2. 新采集的 schema6 使用新输出目录、新 manifest 和新 collector SHA。
3. schema5 可继续训练已经通过现有门控的事件/持续时间头；object head 仍关闭。
4. schema6 数据达到独立 group 支持量后，预注册 group-level OOF，再训练 object head。
5. 只有 object delta 在未参与阈值选择的 holdout 上同时改善误差、校准和任务成功率，才允许
   激活；合成测试通过不构成该实证结论。

## 当前验证范围

`tests/test_etsf_schema6_pose_quality.py` 覆盖纯合成正常轨迹、所有关键异常原因、reset/teleport
区间掩码、HDF 往返、训练读取以及 pose/quality/frame/registry/payload-SHA 篡改。测试不读取现有
轨迹、Fresh、远端文件，也不运行 simulator 或训练。
