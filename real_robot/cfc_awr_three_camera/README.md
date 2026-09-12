# Three-camera CfC-AWR

这是水果和锅盖任务正式三相机 CfC-AWR 后训练代码的 HPC 快照，来自：

```text
/data/user/leviccdong/EKSF/experiments/
cfc_awr_three_camera_fruit_pot_10000_20260911
```

## 文件

- `code/convert_s1_hdf5_to_lerobot.py`：将 S1 HDF5 转为三相机 LeRobot v3；
- `code/preflight_three_camera_cfc_awr.py`：验证源模型、数据、CfC、AWR sidecar 和三相机 ABI；
- `code/train_weighted_preserve_source.py`：运行加权 flow-matching，并逐字节保留源 processor；
- `code/verify_three_camera_actor.py`：验证 checkpoint 计划、模型更新和部署契约；
- `prepare_fruit_three_camera.slurm`：水果三相机数据转换；
- `train_three_camera_cfc_awr.slurm`：水果/锅盖 smoke 与正式训练；
- `receipt_*_10000_*.json`：已完成正式模型的验收记录。

脚本中的绝对路径和 SHA256 是原实验的审计绑定。复跑原实验可以保持不变；更换源 checkpoint 时必须同时更新源路径、SHA、输出目录和对应数据契约。
