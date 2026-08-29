# RoboTwin2 `move_can_pot` 五本体 actor episode 流式 staging v1

## 结论

`scripts/stage_robotwin2_move_can_pot_actor_episodes_v1.py` 把已经完成整包 SHA-256
核验的公开 RoboTwin2 ZIP，逐 episode 物化为 native actor 后续训练可消费的 raw staging。
默认只处理五本体各 50 条 `clean_50`，即 250 条；显式加入 `randomized` 后再处理五本体各
500 条，共 2,750 条。

这不是 critic 监督生成器。公开 expert episode 不会被自动标成 success，也不会产生 failure、
recovery、object-change、event 或 task-success 标签。共享 critic 的这些监督仍必须来自 source-body
冻结 actor 的 simulator rollout/candidate branch，并保持 held-out body 隔离。

## 输入绑定与读取边界

命令必须同时接收：

- 数据盲五本体 LOBO preregistration JSON，逻辑 SHA 固定为
  `75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee`；
- 11 个官方 ZIP 已逐字节核验的 materialization receipt；
- 下载根。下载根可以搬迁，不依赖 `/home/user/...` 固定路径，但选中的 archive 会重新核对
  receipt 中的相对路径、size 与整包 SHA-256。

每个 archive 先核对完整 episode 配对集合：`HDF5 + instruction JSON + mp4 + pkl`。随后只打开
HDF5 与 instruction JSON：

1. 单个 HDF5 从 ZIP 流到 staging 内的受控临时文件；
2. 检查 HDF5 只有 hard links、无 object/vlen dtype，所有 dataset 第一维为同一个帧数；
3. 检查 endpose 为 `[T,7]`、joint components 与 `joint_action/vector` 一致；
4. 检查 instruction JSON 恰好是 `seen/unseen` 两个各 100 条非空字符串的列表；
5. HDF5 与 JSON 以 `0444` create-new 文件发布，最后整个 staging tree 变成只读；
6. `.pkl` 和 `.mp4` 只核对 central-directory 成员名，打开数始终为 0。

HDF5 审计只读取 group/dataset metadata，不读取任何 dataset value，也不解码 JPEG。没有调用
`extract/extractall`，因此不是全量解压。

## 五本体动作维度与统一 14D effect

官方 raw HDF5 的 joint vector 并不完全同维：

| body | `joint_action/vector` |
|---|---:|
| Aloha-AgileX | 14 |
| ARX-X5 | 14 |
| Franka | 16 |
| Piper | 14 |
| UR5 | 14 |

staging 保留原始维度，绝不把 Franka 裁成 14D。真正跨本体的 action 输入由
`scripts/robotwin2_cross_body_canonical_adapter_v1.py` 的
`task_space_action_effect14(...)` 产生。它使用相邻帧的
`endpose/{left,right}_endpose` 与 gripper，按下列固定顺序输出 `[T-1,14]`：

```text
left  [Δx, Δy, Δz, Δaxis_angle_x, Δaxis_angle_y, Δaxis_angle_z, Δgripper]
right [Δx, Δy, Δz, Δaxis_angle_x, Δaxis_angle_y, Δaxis_angle_z, Δgripper]
```

pose 约定为 `[x,y,z,qw,qx,qy,qz]`；平移差在 source world frame；旋转为
`q_next * conjugate(q_current)` 的最短 axis-angle。这个变换无参数、无需用任何本体数据拟合。
staging 阶段不读取数组，所以只在 manifest 中绑定 adapter 文件 SHA 与接口 SHA，不提前物化 14D
数组；后续 actor converter 显式调用该接口。

## 新共享 critic 27D state 接口

同一 adapter 的 `pack_shared_critic_state27(...)` 冻结下列顺序：

```text
relative_goal3
+ left_ee_to_object3 + right_ee_to_object3
+ object_displacement3
+ left/right grippers2
+ object_quaternion_wxyz4
+ event_onehot [e0,e12,e3,e4,eK]5
+ predicates4
= 27D
```

该函数只打包调用方已经提供的 tensor，不从 expert HDF 推断 object、goal、event 或 predicate。
因此当前 raw expert staging 不会声称已经生成 27D critic state，更不会借此伪造 critic 标签。

## 输出布局与 XPolicyLab

输出目录采用 XPolicyLab discovery 易接入的三层布局：

```text
OUTPUT/
├── actor_staging_manifest.json
└── data/RoboTwin2_move_can_pot/move_can_pot/
    ├── aloha-agilex_clean/
    │   ├── data/episode0.hdf5
    │   └── instructions/episode0.json
    ├── arx-x5_clean/
    ├── franka_clean/
    ├── piper_clean/
    └── ur5_clean/
```

需要明确：下载包内 HDF5 是 RoboTwin2 legacy
`observation/joint_action/endpose` schema；通用 XPolicyLab LeRobot v3 converter 当前期望
`vision/state/action` v1.0 schema。因此目录发现形式已经对齐，但不能直接声称 generic converter
ready。下一步应做一个调用上述 14D 接口的 label-free schema adapter，再交给官方 converter；
不得为了“直接可用”而读取 `.pkl` 或暗中选择 instruction。

## 运行

只 staging 完整五本体 clean 250 episodes：

```bash
python scripts/stage_robotwin2_move_can_pot_actor_episodes_v1.py \
  --preregistration /PUBLIC/CONTRACT/robotwin2_lobo_prereg.json \
  --materialization-receipt /PUBLIC/CONTRACT/materialization_receipt.json \
  --download-root /PUBLIC/SNAPSHOT \
  --conditions clean \
  --output /NEW/actor_staging_clean250
```

加入 randomized 2,500 episodes：

```bash
python scripts/stage_robotwin2_move_can_pot_actor_episodes_v1.py \
  --preregistration /PUBLIC/CONTRACT/robotwin2_lobo_prereg.json \
  --materialization-receipt /PUBLIC/CONTRACT/materialization_receipt.json \
  --download-root /PUBLIC/SNAPSHOT \
  --conditions clean randomized \
  --output /NEW/actor_staging_clean_randomized2750
```

`OUTPUT` 必须不存在。成功后 manifest 记录每个 archive、episode、HDF/JSON SHA、帧数、raw
action dim、结构 schema SHA、未打开的 pkl/video 成员，以及 adapter 接口绑定。该 staging 完成只
表示 actor 原始输入已经准备好；训练、模拟器评估、checkpoint promotion 与跨本体成功率结论不由
这个 manifest 授权。
