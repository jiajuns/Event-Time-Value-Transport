# ETSF V8 Release 资产说明

本文档说明 GitHub Release `v8.0.0-icra` 中每个归档的用途、输入契约和复核方式。Release 只分发正式 V8 仿真 checkpoint、UMI/CfC 价值模块、三个三相机 CfC-AWR 策略以及论文绘图所用的训练结果；原始视频、HDF5、缓存、一次性测试脚本和 Event V4 策略不在发布范围内。

## 下载与校验

```bash
gh release download v8.0.0-icra \
  --repo jiajuns/Event-Time-Value-Transport \
  --dir etsf-v8-release

cd etsf-v8-release
sha256sum -c SHA256SUMS
tar --use-compress-program=unzstd -xf ASSET_NAME.tar.zst
```

## 资产清单

| Release asset | 内容 | 用途 |
|---|---|---|
| `etsf-v8-simulation-checkpoints.tar.zst` | 五个 paired seed 的 V8 `step_3000.pt` | 复核仿真 critic 与跨本体指标 |
| `etsf-v8-umi-three-task-critics.tar.zst` | 共享 UMI RGB-CfC、训练回执、蔬菜/水果/锅盖的五折 value adapters | 从真实视频事件表示生成 OOF value/advantage |
| `etsf-v8-cfcawr-vegetable-three-camera.tar.zst` | 完整 SmolVLA `pretrained_model` | 蔬菜放盘子部署 |
| `etsf-v8-cfcawr-fruit-three-camera.tar.zst` | 完整 SmolVLA `pretrained_model` | 水果放盘子部署 |
| `etsf-v8-cfcawr-pot-lid-three-camera.tar.zst` | 完整 SmolVLA `pretrained_model` | 拿起锅盖并放到锅旁部署 |
| `etsf-v8-training-curves-and-results.tar.zst` | V8 五 seed loss、最终指标、审计记录和论文绘图产物 | 复核表格与绘图 |
| `SHA256SUMS` | Release 资产 SHA256 | 下载完整性检查 |

压缩归档的最终大小与 SHA256：

| Asset | Bytes | SHA256 |
|---|---:|---|
| `etsf-v8-simulation-checkpoints.tar.zst` | 1,714,566 | `1d4c9a42924247341ea9aa6e993ded8c77bff98880889f939b76249c065b6c39` |
| `etsf-v8-umi-three-task-critics.tar.zst` | 3,711,988 | `6d19926b96773011f6767f7a8ac66dacca7797838300d76c0da87396ac7638f4` |
| `etsf-v8-cfcawr-vegetable-three-camera.tar.zst` | 721,062,812 | `13db45ee2804cb62dbb67d25a1df610341a8a32eada25de5abbe08162707fdd2` |
| `etsf-v8-cfcawr-fruit-three-camera.tar.zst` | 720,771,572 | `dab5e80f40a618c729387e817df9b71bd938ec2f092c70a86573ecf60d2defdf` |
| `etsf-v8-cfcawr-pot-lid-three-camera.tar.zst` | 721,228,548 | `a8d52a5489f7afe171250364c9646d0639dddafb27b48b6b0d60362b187e018c` |
| `etsf-v8-training-curves-and-results.tar.zst` | 36,129,424 | `0562828d21301a10bdeca81172a3d3a81166f7895b0a7752f92ad446131c536c` |

## 三个策略 checkpoint 的输入契约

三个发布策略均由三相机 SmolVLA checkpoint 经 CfC-AWR 追加 10,000 optimizer steps 得到。部署不加载 critic；critic 只在后训练阶段生成 TD(\(\lambda\)) 优势与样本权重。

| Task | 训练提示词 | 视觉输入 | Proprioception | Action chunk | `model.safetensors` SHA256 |
|---|---|---|---:|---:|---|
| Vegetable | `put the vegetable to the plate` | head + left wrist + right wrist RGB | state25 | \(50\times25\) | `d3738347407382ab89ad0d2bc45a34e82a5375d5c14b40af1dfc9019089681e9` |
| Fruit | `put the fruit to the plate` | head + left wrist + right wrist RGB | state25 | \(50\times25\) | `fea590f488359d3ad7eabf3861f46fd54160cb69c3e152d7088260ae55169fd1` |
| Pot lid | `pick up the pot lid and place it beside the pot` | head + left wrist + right wrist RGB | state25 | \(50\times25\) | `0be5efe0b21a5293644ae973e600d1d213621bf549bc91c24d5679aaa0cf4dc4` |

不要只复制 `model.safetensors`。每个 `pretrained_model` 目录中的配置、归一化和反归一化 processors 都属于部署契约。推理时的任务文本、相机键名、state/action 维度必须与表中一致。

## V8 仿真 checkpoint

归档包含以下五个正式 seed：

| Seed | File | SHA256 |
|---:|---|---|
| 20260901 | `seed20260901/formal/step_3000.pt` | `90f8e7263bed20c4c390ea9270d1dbbca63a10d4e8eaf11f58aa902615397124` |
| 20260902 | `seed20260902/formal/step_3000.pt` | `3f587b4efb26950251f15f50848ec1e32b22317b7856214a387cbb77693ef5eb` |
| 20260903 | `seed20260903/formal/step_3000.pt` | `d7796de76e7cf5dc76fa5a052295092112469b114cffbb54e00cbc671ff05735` |
| 20260904 | `seed20260904/formal/step_3000.pt` | `d9f4516dea800f247060700d43199b45a0c94878157c02d81da3f9788bcd9ee6` |
| 20260905 | `seed20260905/formal/step_3000.pt` | `f1390692c239a9cf6d467b51b8eede6567842a307d30ee427736a74b9190ea58` |

这些 checkpoint 对应仓库 `experiments/experiment2_ablation_fixed_20260901/` 中冻结的训练与评估协议。数据划分和指标应以 `FROZEN_PROTOCOL.json` 与 `paper_results/v8/` 为准。

## UMI/CfC 价值资产

该归档分成一份共享观察器和三个任务适配目录：

```text
shared_umi_rgb_cfc/
tasks/
  vegetable/
  fruit/
  pot_lid/
```

- `shared_umi_rgb_cfc/selected_observer.pt` 是在 UMI 视频事件数据上适配的 RGB-CfC 观察器；S1 state25 不进入该共享主干。
- 每个任务目录包含 `value_all.pt`、五个 episode-level folds、事件权重或 sidecar 以及训练回执。
- 水果和锅盖目录中的 value adapters 使用对应任务的真实机器人轨迹特征拟合；它们共享 UMI 适配的事件观察接口，但不是把同一批 UMI 轨迹重复命名为三个任务。
- 五折输出用于给每条训练轨迹生成 out-of-fold value/advantage；`value_all.pt` 用于完整数据拟合后的分析或新轨迹评分。

## 版本边界

- 发布的是论文对应的 V8/CfC-AWR 主线，不包含 Event V4-AWR 策略 checkpoint。
- Release 中的 SmolVLA 是三相机版本，不是早期单右腕版本。
- Git 仓库提供训练、评估、预检和验收代码；大模型只放在 GitHub Release，不进入 Git 历史。
- 原始真实机器人视频及 HDF5 受体积与数据管理限制，不随公开 Release 分发。

最终资产级 SHA256 与字节数在上传后写入同目录下的 `SHA256SUMS`，以该文件为下载校验依据。
