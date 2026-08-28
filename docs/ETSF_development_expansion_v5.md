# Schema-v5 development expansion（150 scenes）

该数据仅用于模型开发，永远不能作为 fresh confirmation。候选范围冻结在
`artifacts/protocol/development_expansion_seed_candidates_20260827.json`：从
`100100276` 开始连续 200 个 requested seeds。预注册脚本只执行 reset identity，按冻结
顺序选择前 150 个 resolved-unique scenes，并排除 official150 与 frozen fresh50 的
requested/resolved 集合；不加载策略，不执行动作，不读取 event、reward 或 success。

先生成冻结 development manifest：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/preregister_robotwin_development_expansion_seeds.py \
  --candidates /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/development_expansion_seed_candidates_20260827.json \
  --fresh-seed-manifest /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/fresh_confirmation_seeds_20260827.json \
  --output /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/development_expansion_seeds_20260827.json \
  --rlinf-root /home/user/etsf_stage0/RLinf \
  --robotwin-root /home/user/etsf_stage0/RoboTwin \
  --robotwin-code /home/user/etsf_stage0/RoboTwin_RLinf_support
```

随后先给 launcher 加 `--dry-run` 做完整 SHA/overlap/argv 审计，再正式运行。collector
固定使用 `--seeds-key train --allow-unregistered-seeds
--development-seed-manifest ...`，落盘 registry 为
`explicit_development_expansion`，fresh manifest 字段必须为空。候选固定为 deterministic
加 `.25/.5/.75/1.0` 四个 blend（共 5 个），temperature `.7`、top-k `4`、保留 gripper。
部分采集中断仅允许相同冻结 contract 显式 `--resume`。

采集完成后、任何 250-group 训练前必须生成一次新的 CPU 审计：

```bash
/home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/audit_openvla_etsf_development250.py \
  --old-data /home/user/etsf_openvla_event_branches_v5_train100_20260827 \
  --development-data /home/user/etsf_openvla_event_branches_v5_development150_20260827 \
  --development-seed-manifest /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/development_expansion_seeds_20260827.json \
  --fresh-seed-manifest /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/fresh_confirmation_seeds_20260827.json \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --output /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/development250_data_audit_20260827.json
```

审计只有在 old100 + development150 恰好 250 groups、共 1150 candidate branches，
新数据严格 5 candidates/`explicit_development_expansion`/fresh 字段为空，并且 new150
与 old100/fresh50 resolved identity 零重叠时才写出 `training_ready`。报告同时冻结每个
HDF5 SHA、success、组内 outcome variation、duration/reached-event observed、event 分布、
continuation query 数和 terminal steps 等标签密度；已有输出拒绝覆盖。

审计等待器可与下面的服务器流水线同时提前用 `nohup` 启动。流水线不会轮询或读取采集中
HDF5，只等待审计器原子发布签名的 `training_ready` JSON：

```bash
nohup /home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  /home/user/etsf_event_world_model_code_20260827/scripts/launch_openvla_etsf_development250_oof_pipeline.py \
  --audit /home/user/etsf_event_world_model_code_20260827/artifacts/protocol/development250_data_audit_20260827.json \
  --old-development100 /home/user/etsf_openvla_event_branches_v5_train100_20260827 \
  --new-development150 /home/user/etsf_openvla_event_branches_v5_development150_20260827 \
  --merged-output /home/user/etsf_openvla_event_branches_v5_development250_20260827 \
  --oof-output /home/user/etsf_openvla_counterfactual_oof_v5_development250_actionrank_20260827 \
  --state-root /home/user/etsf_development250_oof_pipeline_state_20260827 \
  --pretrained /home/user/etsf_openvla_structured_event_world_model_move_can_pot_sealed_schema3_20260827/seed_20260827/event_world_model_best.pt \
  --event-spec /home/user/etsf_stage2_run_20260825/event_spec.json \
  --python-bin /home/user/etsf_stage0/RLinf/.venv_openvla_robotwin/bin/python \
  > /home/user/etsf_development250_oof_pipeline_20260827.log 2>&1 < /dev/null &
```

它验证审计签名和源路径后，以 hard link 原子建立异构候选数的 development250（100×4 +
150×5），确认 RTX4090 无并发计算再串行运行 5×50 OOF。未授权时状态固定为
`complete_guard_not_authorized_fresh_forbidden`；只有授权且完成全250 refit 才输出
`complete_fresh50_ready_one_shot`。本脚本不接受任何 fresh 数据参数。
