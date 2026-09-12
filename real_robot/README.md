# Real-robot training code

本目录保存论文三任务真实机器人验证所需的正式代码入口。模型权重不进入 Git 历史，在 GitHub Release 中发布。

## 目录

- `cfc_awr_three_camera/`：水果与锅盖三相机 SmolVLA 的 CfC-AWR 10,000-step 后训练；
- `../umi_cfc/_internal/cfc_value_umi_s1_20260906/`：蔬菜任务的 UMI RGB-CfC、state25 独立 value adapter 与 CfC-AWR；
- `../umi_cfc/_internal/rl_fruit_event_v4_cfc_awr_10000_20260909/`：水果轨迹的冻结 RGB-CfC 特征、五折 value 与 AWR sidecar；
- `../umi_cfc/_internal/pot_lid_event_v4_cfc_awr_10000_20260910/`：锅盖轨迹的冻结 RGB-CfC 特征、五折 value 与 AWR sidecar；
- `../umi_cfc/_internal/finetune_v4/train_event_weighted.py`：按全局数据 index 注入权重，计算 `mean(weight × per-sample flow loss)`。

## 三个正式策略

| Task | Actor input | Prompt | Release asset |
|---|---|---|---|
| Vegetable placement | 3 RGB + state25 | `put the vegetable to the plate` | `etsf-v8-cfcawr-vegetable-three-camera.tar.zst` |
| Fruit placement | 3 RGB + state25 | `put the fruit to the plate` | `etsf-v8-cfcawr-fruit-three-camera.tar.zst` |
| Pot-lid relocation | 3 RGB + state25 | `pick up the pot lid and place it beside the pot` | `etsf-v8-cfcawr-pot-lid-three-camera.tar.zst` |

三个 checkpoint 均输出 `50 × 25` 连续动作。CfC/value 模块只在离线评分和后训练阶段使用，部署只加载完整 SmolVLA `pretrained_model` 目录。

## 水果/锅盖训练

HPC 原始路径写在 Slurm 文件中，使用前按实际服务器更新 `ROOT`。先运行 5-step smoke，检查 receipt 为 `passed` 后再运行正式 10,000-step 作业：

```bash
cd real_robot/cfc_awr_three_camera

sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=5 train_three_camera_cfc_awr.slurm
sbatch --export=ALL,CFC3_TASK=pot,CFC3_STEPS=5 train_three_camera_cfc_awr.slurm

sbatch --export=ALL,CFC3_TASK=fruit,CFC3_STEPS=10000 train_three_camera_cfc_awr.slurm
sbatch --export=ALL,CFC3_TASK=pot,CFC3_STEPS=10000 train_three_camera_cfc_awr.slurm
```

正式作业使用学习率 `1e-6`、batch size 32、100-step warmup、cosine decay，并每 1000 步保存。preflight 会拒绝相机集合、state/action 维度、任务文本、源模型 SHA、sidecar 身份或 chunk 边界不匹配的输入。

## 蔬菜训练

```bash
cd umi_cfc/_internal/cfc_value_umi_s1_20260906
sbatch train_hpc.slurm
```

该入口同样绑定原实验路径和 SHA。替换数据或纯 SmolVLA checkpoint 时必须建立新的实验目录并更新 preflight，不要覆盖已验收实验。

## 数据接口

- 图像键：`observation.images.base_0_rgb`、`observation.images.left_wrist_0_rgb`、`observation.images.right_wrist_0_rgb`；
- 状态：`observation.state`，25 维；
- 动作：`action`，25 维；
- `chunk_size = n_action_steps = 50`；
- LeRobot v3 数据必须具有连续、唯一的全局 `index`。

三相机策略训练读取全部三路图像。价值观察阶段以头部全局视角为主；state25 只进入独立 adapter，不进入共享 UMI RGB-CfC 编码器。
