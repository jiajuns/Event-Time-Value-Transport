# RoboTwin2 五本体 EE16 actor 转换

`scripts/convert_robotwin2_staging_to_xpolicylab_ee16_v1.py` 解决的是当前效果路径里的
actor 动作语义不统一问题：Aloha、ARX、Franka、Piper、UR5 不再把不同维数的关节向量混到
同一个策略里，而是统一为双臂绝对末端位姿与夹爪：

```text
left  [x, y, z, qw, qx, qy, qz, gripper]
right [x, y, z, qw, qx, qy, qz, gripper]
= 16D
```

state 与 action 均为这个 EE16；默认 `action[t] = EE16[t+1]`，最后一帧丢弃。每条轨迹的
四元数会归一化并消除 `q/-q` 符号跳变，避免同一姿态形成互相冲突的监督。三个训练相机固定为
head、left wrist、right wrist，公开 JSON 的 seen 指令按 episode 确定性轮换，频率默认采用官方
task 的 15Hz。

输出同时包含语义正确的 `left/right_ee_poses`，以及当前
`transform_lerobot_v30_format.py` 所需的同数据 hard-link 兼容字段；不是把 EE pose 当成关节重新
计算。脚本不会读取 `.pkl`，也不会生成 critic、event、成功/失败/恢复或 object-effect 标签。

## 4090 远程完整转换

下面一次读取已经完整完成的 clean250 和 randomized2500 两棵 staging，共转换五本体 2,750
条轨迹。命令中的转换器是当前已逐字节同步到 4090 的只读副本，SHA-256 为
`fa19e09613f12eabd95b391d5164ad94f028fff9405159c28586c198d5008aee`。

```bash
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  /home/user/etsf_robotwin2_ee16_converter_code_precommit_20260830_v1/convert_robotwin2_staging_to_xpolicylab_ee16_v1.py \
  --input-root /home/user/public_actor_staging/robotwin2_move_can_pot_5emb_clean250_a967b852_20260830_v1 \
  --input-root /home/user/public_actor_staging/robotwin2_move_can_pot_5emb_randomized2500_a967b852_20260830_v1 \
  --output-root /home/user/etsf_stage0/RoboTwin/data \
  --dataset-name RoboTwin2_move_can_pot_EE16 \
  --task move_can_pot \
  --env-cfg-type etsf_ee16_15hz \
  --frequency 15 \
  --action-alignment next \
  --instruction-split seen \
  --instruction-index -1 \
  --xpolicylab-project-root /home/user/etsf_stage0/RoboTwin
```

它会生成十个 XPolicy target（五本体乘 clean/randomized）并安装一个很小的
`env_cfg/etsf_ee16_15hz.yml`。该配置只让官方 converter 知道数据是每臂 `7+1` 和 15Hz；
`dual_franka` 在这里仅作为既有维数描述，不把任一来源本体声明为 Franka。

随后直接运行现有 XPolicyLab LeRobot v3 converter：

```bash
cd /home/user/etsf_stage0/RoboTwin
/home/user/anaconda3/envs/ETSF_RoboTwin/bin/python \
  XPolicyLab/scripts/transform_lerobot_v30_format.py \
  'RoboTwin2_move_can_pot_EE16.*.etsf_ee16_15hz' \
  --repo_id robotwin2_move_can_pot_5emb_ee16_full2750 \
  --data_type RoboDojo \
  --data_version v1.0 \
  --max_episode 500 \
  --resolution 240x320
```

如果要先训练五个 native actor，不要把五本体混成一个模型；分别把 pattern 限定到对应的
`move_can_pot__<body>_clean` 与 `move_can_pot__<body>_randomized` 两个 target，再为每个 body
生成独立 LeRobot repo。共享事件头的 LOBO 实验只在 critic 监督上留出目标本体；native actor
是否用目标本体专家数据必须单独标注，不能把它写成整套策略零样本迁移。
