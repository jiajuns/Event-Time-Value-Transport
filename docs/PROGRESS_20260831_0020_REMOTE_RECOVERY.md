# 远程共享头流水线恢复加固（2026-08-31 00:20 CST）

## 4090 当前状态

- 正式任务仅运行在 `user@100.115.128.14` 的 RTX 4090
  `GPU-06f6e50e-5296-258f-dd86-8f838390a7d1`；本机未启动训练或仿真。
- 采集快照为 587/2000 decisions、2348/8000 branches；Aloha/ARX/Franka/Piper/UR5
  分别为 99/128/106/128/126 decisions。
- 三路 collector 持续产出，GPU 100%，约 18.3/24.6 GiB，无 CUDA OOM。
- 当前吞吐约 37--45 decisions/hour，正式采集还需约 32--38 小时；随后才会进入
  C-only LOBO、正式 N=4 paired 与完整消融。

## 当前数据说明了什么

- 文件、哈希、shape、finite、mask 与候选动作差异均通过只读审计。
- 当前只有 1/587 个成功决策，且该决策四个候选全部成功；candidate0 与 N=4 oracle
  成功率均为 0.170%，尚无 action-conditioned binary-success 排序监督。
- e3/e4 与 recovery 监督当前均为零；253/587 个决策仍提供了全失败条件下的 dense
  进展/避灾排序信号。
- 因此 C 主流数据继续完整采集，但不能把其内部 critic 指标当作跨本体成功证据。
  source-train-only 的 e3/e4 expert-root 补充与 held-out paired `Delta SR` 仍是必要环节。

## 本轮发现并修复的无人值守风险

1. 正式采集日志中存在少量 RoboTwin `UnStableError`。旧 watcher 会在全部已排队任务
   结束后传播任一失败 future，可能无法自动进入缺口补采。修正版会记录失败 block，继续
   其余 block，并用不进入正式评测 seed 区间的 supplemental seed 补足每个 stratum；连续
   8 个 supplemental seed 均失败才判定为系统性故障。
2. C+B upgrade watcher 的运行环境缺少
   `/home/user/anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages`，会导致首个 B
   collector 导入 `safetensors` 失败。修正版将该路径作为显式、存在性校验的 CLI 输入，
   不再依赖 ambient `PYTHONPATH`。
3. 当前 v8 正式采集不被中断；部署修正版后由独立 CPU guardian 在旧 watcher 自然退出且
   数据未完成时续跑。C+B watcher可立即切换到修正版，因为它目前只在等待上游、不占 GPU。

最终完成标准不变：五个 held-out 本体完整报告 N=4/N=8 的 paired `Delta SR`、阶段进度、
95% CI 与 McNemar；在这些真实 rollout 完成前不得宣称跨本体成功率已提高。
