# RoboTwin2 `move_can_pot` 五本体公开下载物化校验器 v1

`scripts/verify_robotwin2_move_can_pot_public_materialization_v1.py` 只回答一个问题：本地下载根中的公开 `dataset/move_can_pot` 是否与五本体 LOBO 预注册冻结的官方切片逐字节一致。

校验器要求该任务目录恰好包含 11 个官方 ZIP，逐项核对相对路径、字节数和 LFS payload SHA-256（总计 21,238,835,871 bytes）。它还只读 ZIP central directory，审计成员路径、重复名、加密标志、CRC 字段和压缩元数据。它不会解压，不会读取成员 payload，不会反序列化 pickle、NumPy 或 Torch 数据，也不会解码图像/视频。下载根外层的 Hugging Face 缓存与运行日志不属于官方任务目录，不参与“额外文件”判定。

用法：

```bash
python scripts/verify_robotwin2_move_can_pot_public_materialization_v1.py \
  --download-root /path/to/public_snapshot \
  --preregistration /path/to/robotwin2_lobo_prereg.json \
  --output /path/to/new/materialization_receipt.json
```

省略 `--preregistration` 时会从已审查模块确定性重建同一预注册，并执行完整一致性验证。只有文件集合、全部 size、全部 payload SHA-256 和 ZIP 中央目录安全检查均通过，才会以 create-once、`0444`、单硬链接形式生成 `materialized=true` 的内容寻址收据；任一失败都在写收据前终止。收据同时绑定 verifier 与预注册实现文件本身的 SHA-256，避免以后用改过的校验逻辑重新解释旧收据。

该收据只证明公开下载完整。它不验证样本语义、标签或 episode 数，不授权训练、模拟器运行、评估、模型选择、部署或跨本体性能声明；这些动作仍需后续独立权限与协议。
