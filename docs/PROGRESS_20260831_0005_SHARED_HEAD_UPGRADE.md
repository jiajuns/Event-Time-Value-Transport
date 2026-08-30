# 共享头增强进度（2026-08-31 00:05 CST）

## 当前远程任务

- 正式工作继续运行在 `user@100.115.128.14` 的 RTX 4090；本机只做代码测试。
- GPU：`GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`，记录时 76%，14239/24564 MiB。
- 正式 C 数据：580/2000 decisions，2320/8000 branches。
- 分本体：Aloha 97、ARX 128、Franka 104、Piper 128、UR5 123 decisions。
- C-only LOBO watcher：`waiting_for_complete_public_branches`。
- 正式 N=4 paired watcher：`waiting_for_true_complete_five_fold_lobo`。
- 完整消融 watcher：`waiting_for_formal_paired_completion`。
- 当前仍是正式数据采集，不是共享头训练；没有中断或替换任何远程进程。

## 本轮完成的代码

- 完整 e3/e4 scripted-root/frozen-actor 补充 collector（100 decisions / 400 branches）。
- 五本体 supplement binding materializer。
- trainer 的 C+B source-only proper-world stream，固定 `lambda=0.25`，held-out supplement zero-open。
- 独立 N=8/N=16 actor-flow candidate-pool paired runner。
- N=4/N=8 fold-regime binding：增强评估必须使用同一个 supplement binding SHA，拒绝 C-only/增强混折。
- 原 LOBO watcher支持独立 C+supplement 五折训练，不改变现有 C-only watcher。
- 完整增强 watcher：等当前 C-only collection→LOBO→paired→ablation 全部结束后，自动运行
  B collection→binding→C+B LOBO→完整 N=4 paired→完整 N=8 paired。
- 最新详细设计：`docs/ETSF_ROBOTWIN2_SHARED_HEAD_V8_CROSS_EMBODIMENT_UPGRADE_ZH.md`。

## 验证

- 联合覆盖 trainer、两类 collector、materializer、N=4/N=8 runner、评估器、LOBO/paired/ablation watcher
  和增强 watcher。
- 结果：`143 passed, 1 warning in 7.54s`。
- warning 仅为本机 PyTorch/CUDA 驱动版本提示；正式 GPU 任务固定在远程 4090。

## 尚未完成

- 本轮改动仍需最后 diff 审计、commit、push 和远程只读部署。
- 新增强 watcher尚未挂到远程；挂载时只能等待现有正式 pipeline 完成，不得与当前 collector 抢 GPU。
- 真实跨本体效果仍未知。完成条件是 held-out 五本体完整 N=4/N=8 配对 `Delta SR`、阶段进度、95% CI
  和 McNemar，而不是上述测试通过。
