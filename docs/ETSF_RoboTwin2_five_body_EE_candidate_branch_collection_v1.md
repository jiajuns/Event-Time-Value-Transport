# RoboTwin2 五本体 EE 候选分支采集 v1

入口：`scripts/collect_robotwin2_five_body_ee_candidate_branches_v1.py`。

该入口为共享事件头生成真正的 critic 监督，而不是把官方 expert 正样本冒充成成败数据。每个本体
使用冻结的 EE16 SmolVLA actor；对 clean/randomized 的独立开发 seed，在固定 query 0/1/2/3 处从
同一 actor、同一状态和固定 flow-noise 规则采四个候选。candidate 0 是 actor baseline。每个候选都
从相同 seed 重置并完整重放相同 prefix，然后执行首段动作并由同一个 actor继续到终止或 200 step。

模拟器逐步记录 can/pot 真值 pose、左右 EE 与夹爪，随后生成一个四行 NPZ：

- 27D 本体无关状态；
- EE SE(3)+gripper 的 14D action effect；
- post/next event、物理秒 duration；
- success、regression/recovery；
- moving object 与 relative-goal 的 6D 变化；
- 固定 candidate index 0–3 与实际首段 elapsed seconds。

开发采集默认 seed 从 `2026081000` 开始，与正式配对评估的
`2026090000..2026090099` 完全分开。manifest 只保存 group identity、condition、seed、query、路径和
payload SHA，不复制 candidate success/event 标签；因此外层 held-out body 的 manifest 可用于 LOBO
分配而不提前读取目标 outcome。

默认规模为 50 seed × 4 query × 2 condition × 5 body = 2,000 同根决策，即 8,000 条完整候选分支。
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
export COLLECTOR_CODE=/home/user/etsf_robotwin2_fivebody_branch_collector_dev_20260830
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
        task.take_action(action, action_type="ee")
        assert action.shape == (16,) and np.isfinite(action).all()
        assert all(shape == (240, 320, 3) for shape in rgb)
        print(body, BODY_EMBODIMENT[body], rgb, "ee16/action/success=OK", bool(task.check_success()))
    finally:
        task.close_env(clear_cache=False)
PY
```

## 五本体顺序采集命令

当前远程机尚无任何 16D EE SmolVLA actor checkpoint。已有 checkpoint 是 14D joint actor，collector
会按设计拒绝它，不能伪装成 EE actor。等五个 EE16 actor 目录生成后，按下面固定路径放置并顺序采集；
每个进程都写独立日志和 body 目录，可以安全断开 SSH：

```bash
export ACTOR_ROOT=/home/user/etsf_robotwin2_ee16_actors
export BRANCH_ROOT=/home/user/etsf_robotwin2_fivebody_branches_20260830
export VLM_METADATA=/home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata
export EVENT_SPEC=/home/user/etsf_schema6_event_spec_r6g_20260828.json
mkdir -p ${BRANCH_ROOT}/logs

nohup bash -lc '
set -euo pipefail
export PYTHONNOUSERSITE=1
export COLLECTOR_CODE=/home/user/etsf_robotwin2_fivebody_branch_collector_dev_20260830
export ROBOTWIN_ROOT=/home/user/etsf_stage0/RoboTwin
export PYTHONPATH=${COLLECTOR_CODE}:/home/user/etsf_stage0/lerobot/src:/home/user/etsf_stage0/.venv_lerobot_smolvla_v044/lib/python3.10/site-packages:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/envs/curobo/src:/home/user/anaconda3/envs/ETSF_RoboTwin/lib/python3.10/site-packages
export ASSETS_PATH=${ROBOTWIN_ROOT}
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/user/anaconda3/envs/RoboTwin2/bin/python
ACTOR_ROOT=/home/user/etsf_robotwin2_ee16_actors
BRANCH_ROOT=/home/user/etsf_robotwin2_fivebody_branches_20260830
VLM_METADATA=/home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata
EVENT_SPEC=/home/user/etsf_schema6_event_spec_r6g_20260828.json
cd ${ROBOTWIN_ROOT}
for body in aloha-agilex arx-x5 franka piper ur5; do
  ${PY} ${COLLECTOR_CODE}/collect_robotwin2_five_body_ee_candidate_branches_v1.py \
    --body ${body} \
    --actor-checkpoint ${ACTOR_ROOT}/${body}/checkpoints/last/pretrained_model \
    --vlm-metadata-path ${VLM_METADATA} \
    --robotwin-root ${ROBOTWIN_ROOT} \
    --event-spec ${EVENT_SPEC} \
    --conditions clean randomized \
    --seed-start 2026081000 \
    --seed-count 50 \
    --root-query-indices 0 1 2 3 \
    --action-exec-steps 5 \
    --max-steps 200 \
    --output ${BRANCH_ROOT}/${body} \
    > ${BRANCH_ROOT}/logs/${body}.log 2>&1
done
' > ${BRANCH_ROOT}/logs/launcher.log 2>&1 &
echo $! > ${BRANCH_ROOT}/launcher.pid
```

单本体前台调用形式为：

```bash
${PY} ${COLLECTOR_CODE}/collect_robotwin2_five_body_ee_candidate_branches_v1.py \
  --body piper \
  --actor-checkpoint /home/user/etsf_robotwin2_ee16_actors/piper/checkpoints/last/pretrained_model \
  --vlm-metadata-path /home/user/etsf_stage0/offline_assets/smolvlm2_500m_metadata \
  --robotwin-root /home/user/etsf_stage0/RoboTwin \
  --event-spec /home/user/etsf_schema6_event_spec_r6g_20260828.json \
  --conditions clean randomized \
  --seed-start 2026081000 \
  --seed-count 50 \
  --output /home/user/etsf_robotwin2_fivebody_branches_20260830/piper
```

该阶段的输出用于训练，不是最终测试结果。论文主数字仍来自五个 held-out body 上冻结策略
candidate 0 与 ETSF best-of-4 的 1,000 组配对、2,000 次 rollout。
