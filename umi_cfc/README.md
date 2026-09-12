# UMI RGB-CfC 与 S1 价值适配

本目录保留论文真实机器人阶段所用的最小实现：先把 UMI 腕部 RGB 编码成与本体无关的事件时序表示，再用独立 S1 state25 adapter 拟合标量价值，最后以 episode-level 五折交叉拟合生成 TD(\(\lambda\)) 优势和 Event-AWR 权重。

```text
UMI wrist RGB
  -> frozen RGB encoder + CfC event history (96-D)
  -> four event-head probabilities (15-D)
  -> z_video (111-D)

S1 state history (8 x 25)
  -> S1-only adapter (64-D)

[z_video, g_S1]
  -> scalar V
  -> five-fold OOF TD(lambda) advantage
  -> clipped Event-AWR sidecar
```

共享 RGB-CfC 从不读取 state25、原始动作、任务结局或未来帧。state25 只用于真实 S1 的独立 value adapter，因此不会把 S1 关节含义写回共享事件表示。

## 正式代码

- `_internal/event_rl/`：CfC 与事件观察器实现；
- `_internal/scripts/train_factorized_event_observer_v2.py`：训练 RGB-CfC 观察器；
- `_internal/cfc_value_umi_s1_20260906/pipeline.py`：UMI 预训练与冻结特征提取；
- `_internal/cfc_value_umi_s1_20260906/value.py`：state25 adapter、连续时间折扣、TD(\(\lambda\)) 与五折 OOF 权重；
- `_internal/cfc_value_umi_s1_20260906/train_hpc.slurm`：蔬菜任务 CfC-AWR 入口；
- `_internal/rl_fruit_event_v4_cfc_awr_10000_20260909/`：水果任务的冻结 CfC 特征、value 拟合与审计入口；
- `_internal/pot_lid_event_v4_cfc_awr_10000_20260910/`：锅盖任务的冻结 CfC 特征、value 拟合与审计入口。

后两个目录名保留原实验路径以便核对回执；正式 Release 只分发其中的 **original/frozen CfC-AWR** 分支，不分发 Event V4 策略 checkpoint。

## 训练顺序

```bash
# 1. 只使用 UMI origin 的 train/validation split 训练 RGB-CfC
python umi_cfc/_internal/cfc_value_umi_s1_20260906/pipeline.py pretrain \
  --umi /path/to/origin_cache.npz \
  --output /path/to/pretrain \
  --steps 2000 \
  --device cuda

# 2. 冻结观察器，从 S1 LeRobot v3 数据提取 111-D 因果视频特征
python umi_cfc/_internal/cfc_value_umi_s1_20260906/pipeline.py extract \
  --data /path/to/s1_lerobot_v3 \
  --checkpoint /path/to/selected_observer.pt \
  --output /path/to/s1_features.npz \
  --device cuda

# 3. 用经核验的终局标签拟合五折 value，并写出 event_weights.parquet
python umi_cfc/_internal/cfc_value_umi_s1_20260906/value.py \
  --features /path/to/s1_features.npz \
  --labels /path/to/terminal_labels.json \
  --output /path/to/critic \
  --steps 1500 \
  --device cuda
```

蔬菜 actor 的原实验入口是：

```bash
sbatch umi_cfc/_internal/cfc_value_umi_s1_20260906/train_hpc.slurm
```

水果和锅盖的正式三相机 actor 训练统一使用 `real_robot/cfc_awr_three_camera/train_three_camera_cfc_awr.slurm`。

## 固定协议

- UMI 观察器：2,000 步，85 条 train、21 条 validation，窗口 24 个 10 Hz 观测；
- 视频表示：96-D CfC history + 15-D event probabilities；
- S1 adapter：8 个 state25 历史点映射到 64-D；
- action chunk：50 帧，约 1.67 s；
- 折扣：\(\gamma_t=\exp[-0.1(t_{n(t)}-t)]\)；
- TD：\(\lambda=0.95\)；
- 权重：\(w=\mathrm{clip}(\exp(A/(3\,\mathrm{scale}(A))),0.9,1.1)\)；
- 当前 episode 不由用过该 episode 的 value fold 给自己评分；
- unknown 终局不视为失败，actor 权重保持 1；
- 这是离线加权回归，不进行在线探索或反事实 action-chunk 搜索。

## 权重

Git 仓库不保存 `.pt`、视频、HDF5 或特征缓存。正式共享观察器、三个任务的 `value_all.pt`、五折 value、权重 sidecar和训练回执统一在 GitHub Release 的：

```text
etsf-v8-umi-three-task-critics.tar.zst
```

策略 checkpoint 分成三个独立 Release 资产，详见 [Release 资产说明](../docs/RELEASE_ASSETS.md)。
