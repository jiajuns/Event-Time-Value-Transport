# RoboTwin2 五本体 EE 候选分支采集 v1

入口：`scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py`。

该入口为共享事件头生成真正的 critic 监督，而不是把官方 expert 正样本冒充成成败数据。五个本体
共用同一个冻结的 EE16 SmolVLA actor；对 clean/randomized 的独立开发 seed，在固定
query `0/5/10/15`（约为第 `0/25/50/75` 个动作帧）处从
同一 actor、同一状态和固定 flow-noise 规则采四个候选。candidate 0 是 actor baseline。每个候选都
从相同 seed 重置并完整重放相同 prefix，然后执行首段动作并由同一个 actor继续到终止或 200 step。

模拟器逐步记录 can/pot 真值 pose、左右 EE 与夹爪，随后生成一个四行 NPZ：

- 27D 本体无关状态；
- EE SE(3)+gripper 的 14D action effect；
- post/next event、物理秒 duration；
- success、regression/recovery；
- moving object 与 relative-goal 的 6D 变化；
- 固定 candidate index 0–3 与执行前已知的 planned horizon `dt=5/15s`。

`duration` 与 `dt` 的语义严格分开。RoboTwin 一次 EE action 会执行可变数量的内部物理步；采集器用
透明 scene proxy 计数真实 `scene.step()`，再乘 `scene.get_timestep()`，只把这个物理时间用于事件
边界 `duration`。`dt` 是 critic 打分时已知的五步动作窗口，固定为 `5/15s`，不会泄漏候选执行后的
信息。e4 也不再是“连续三个可变时长动作端点”：它要求近目标且物理速度不超过解析规范中明确先验的
`0.01 m/s`，并在 simulator time 上持续至少 `0.2s`。

五本体正式链只接受
`configs/robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json`，文件 SHA-256 为
`4df5b7242d1c7bf8e3f5dac65c0eb4376043dbf6c60ef2633d086ab06e7e3aee`。该规范没有复用
Stage2 从 Aloha/ARX 成功标签和终点拟合的旧规范：moving=`can`、anchor=`pot`，目标左右侧取
`sign(initial_can_x-initial_pot_x)`，目标为初始 pot 的同侧 `0.18m`。near-goal 半径
`0.02m=min(0.035,0.20-0.18)` 直接来自公开任务代码；移动/抬升 `0.01m`、静止速度
`0.01m/s` 与静止窗 `0.2s` 均明确记录为未拟合先验。采集和在线 paired runner 导入同一个解析实现，
所以 e0/e12/e3/e4/eK 与 state27 的前三个 relative-goal 通道不会各自重写。

开发采集默认 seed 从 `2026081000` 开始，与正式配对评估的
`2026090000..2026090099` 完全分开。manifest 只保存 group identity、condition、seed、query、路径和
payload SHA，不复制 candidate success/event 标签；因此外层 held-out body 的 manifest 可用于 LOBO
分配而不提前读取目标 outcome。

正式规模为每个 body/condition/query 恰好 50 个完整决策，即
`50 × 4 query × 2 condition × 5 body = 2,000` 个同根决策、8,000 条完整候选分支。先运行开发
seed `2026081000..2026081049`；若晚期 query 因 actor 提前成功而缺失，watcher 只用后续开发 seed
在相同 condition/query 分层补足到 50，不把 query 回退到早期，也绝不进入正式评估 seed
`2026090000..2026090099`。
五个本体顺序运行以共享一张 4090。采集完成后，每个 outer LOBO fold 仅打开另外四个 source body
的 NPZ；held-out body payload 不参与 normalization、训练或 checkpoint selection。

## 远程 4090 已验证运行环境

远程机为 `user@100.115.128.14`，RoboTwin2 位于
`/home/user/etsf_stage0/RoboTwin`。下面这组环境组合已经真实完成五种本体的场景创建、三相机读取、
16D EE 状态读取、一次 `action_type="ee"` 动作和 `check_success()` 调用：

| body | RoboTwin embodiment | RGB/EE/action/success |
|---|---|---|
| aloha-agilex | `aloha-agilex` | 通过 |
| arx-x5 | `ARX-X5, ARX-X5, 0.6` | 通过 |
| franka | `franka-panda, franka-panda, 0.8` | 通过 |
| piper | `piper, piper, 0.6` | 通过 |
| ur5 | `ur5-wsg, ur5-wsg, 0.8` | 通过 |

必须使用以下解释器和路径顺序。`PYTHONNOUSERSITE=1` 不能省略；直接使用 v044 venv 会从 user-site
误载入 Torch 2.10/CUDA 12.8，并在 CuRobo 初始化时段错误。以下组合实际加载的是 Torch
2.4.1+cu121、LeRobot 0.4.4 和本地 SmolVLM metadata：

```bash
export PYTHONNOUSERSITE=1
export COLLECTOR_CODE=/home/user/etsf_robotwin2_fivebody_full8000_code_20260830_v2_analytic
export ROBOTWIN_ROOT=/home/user/etsf_stage0/RoboTwin
export PYTHONPATH=${COLLECTOR_CODE}:/home/user/etsf_stage0/lerobot/src:/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/envs/curobo/src:/home/user/anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages
export ASSETS_PATH=${ROBOTWIN_ROOT}
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PY=/home/user/anaconda3/envs/RoboTwin2/bin/python
cd ${ROBOTWIN_ROOT}
```

五本体接口复核命令如下；它不生成训练数据，只验证 collector 使用的真实接口：

```bash
${PY} - <<'PY'
import importlib
import numpy as np
from collect_robotwin2_five_body_ee_candidate_branches_v1 import (
    BODY_EMBODIMENT, _load_task_args, _new_task, current_ee_action16,
)
from pathlib import Path

root = Path("/home/user/etsf_stage0/RoboTwin")
for body in BODY_EMBODIMENT:
    task_class = getattr(importlib.import_module("envs.move_can_pot"), "move_can_pot")
    args = _load_task_args(root, body, "clean")
    args["step_lim"] = 200
    task = _new_task(task_class, args, 2026081000, "Move the can to the side of the pot.")
    try:
        obs = task.get_obs()["observation"]
        rgb = [obs[k]["rgb"].shape for k in ("head_camera", "left_camera", "right_camera")]
        action = current_ee_action16(task)
        before = task.scene.step_count
        task.take_action(action, action_type="ee")
        sim_steps = task.scene.step_count - before
        assert action.shape == (16,) and np.isfinite(action).all()
        assert all(shape == (240, 320, 3) for shape in rgb)
        assert sim_steps > 0 and task.scene.timestep_seconds > 0
        print(body, BODY_EMBODIMENT[body], rgb, "sim_steps", sim_steps,
              "sim_seconds", sim_steps * task.scene.timestep_seconds,
              "ee16/action/success=OK", bool(task.check_success()))
    finally:
        task.close_env(clear_cache=False)
PY
```

## Actor 完成后自动顺序采集

入口 `scripts/watch_robotwin2_ee16_actor_to_five_body_branches_v1.py` 在远程后台等待 actor watcher 的
`status=complete`，等待时不初始化 CUDA。它只接受固定 checkpoint：

`/home/user/etsf_smolvla_models/smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/checkpoints/020000/pretrained_model`

checkpoint 出现后，watcher 先验证并绑定目录树 SHA、`config.json` SHA、state16/action16，再用同一个
actor 顺序运行五个本体。正式输出根固定为：

`/home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic`

完成时额外生成五本体 actor authority、完整 collection receipt 和 LOBO training binding。后台启动：

```bash
CODE=/home/user/etsf_robotwin2_fivebody_full8000_code_20260830_v2_analytic
WATCH=/home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic
nohup /home/user/anaconda3/envs/RoboTwin2/bin/python \
  ${CODE}/watch_robotwin2_ee16_actor_to_five_body_branches_v1.py \
  > ${WATCH}.watcher.log 2>&1 < /dev/null &
echo $! > ${WATCH}.watcher.pid
```

单本体前台调用形式为：

```bash
${PY} ${COLLECTOR_CODE}/collect_robotwin2_five_body_ee_candidate_branches_v1.py \
  --body piper \
  --actor-checkpoint /home/user/etsf_smolvla_models/smolvla_robotwin2_move_can_pot_5emb_ee16_full2750_20k_20260830/checkpoints/020000/pretrained_model \
  --vlm-metadata-path /home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata \
  --robotwin-root /home/user/etsf_stage0/RoboTwin \
  --event-spec /home/user/etsf_robotwin2_move_can_pot_five_body_analytic_event_spec_v1.json \
  --conditions clean randomized \
  --seed-start 2026081000 \
  --seed-count 50 \
  --root-query-indices 0 5 10 15 \
  --output /home/user/etsf_robotwin2_fivebody_ee_candidate_branches_full8000_20260830_v2_analytic/piper
```

该阶段的输出用于训练，不是最终测试结果。论文主数字仍来自五个 held-out body 上冻结策略
candidate 0 与 ETSF best-of-4 的 1,000 组配对、2,000 次 rollout。
