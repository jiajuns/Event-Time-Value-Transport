# scripts 目录索引

本目录保留扁平 Python 模块布局，因为现有 runner、远端只读代码目录和 `PYTHONPATH=scripts` 导入都依赖模块 basename。这里的“分类”是运行职责分类，不通过移动文件破坏已经冻结的协议哈希和导入路径。

## 1. 当前正式主线：RoboTwin2 五本体 v13

以下是当前共享头、RAC、WCM 和 N1/N4/N8 闭环链的生产依赖。处理当前实验时优先只看这一组。

### 数据、事件与 actor 协议

- `robotwin2_move_can_pot_analytic_event_spec_v2.py`
- `robotwin2_cross_body_canonical_adapter_v1.py`
- `robotwin2_actor_execution_protocol_v1.py`
- `bootstrap_robotwin2_stable_roster_to_actor_v25_v1.py`
- `run_robotwin2_five_body_actor_execute5_vs_execute50_v1.py`
- `guard_robotwin2_five_body_actor_execute5_vs_execute50_v1.py`
- `watch_robotwin2_actor_protocol_to_v13_crossbody_v1.py`
- `collect_robotwin2_five_body_ee_candidate_branches_v1.py`
- `collect_robotwin2_scripted_expert_root_actor_branches_v1.py`
- `watch_robotwin2_ee16_actor_to_five_body_branches_v1.py`

### v13 共享事件头

- `train_multibody_canonical_event_world_model.py`
- `train_robotwin2_five_body_lobo_shared_event_head_v1.py`
- `robotwin2_smolvla_shared_event_critic_adapters_v1.py`
- `shared_event_critic_plugin_protocol_v1.py`
- `watch_robotwin2_five_body_branches_to_lobo_training_v1.py`
- `guard_robotwin2_postformal_shared_head_upgrade_v1.py`
- `watch_robotwin2_postformal_shared_head_upgrade_v1.py`

### RAC matched baseline

- `robotwin2_relative_action_critic_adapter_v1.py`
- `train_robotwin2_five_body_lobo_relative_action_critic_v1.py`
- `watch_robotwin2_five_body_branches_to_rac_lobo_training_v1.py`
- `watch_robotwin2_rac_lobo_to_nested_success_v1.py`

### WCM matched baseline

- `robotwin2_wcm_future_latent_baseline_v1.py`
- `robotwin2_wcm_future_latent_adapter_v1.py`
- `train_robotwin2_five_body_lobo_wcm_future_latent_baseline_v1.py`
- `watch_robotwin2_v13_rac_to_wcm_lobo_training_v1.py`
- `watch_robotwin2_wcm_lobo_to_nested_success_v1.py`

### 闭环、oracle 与正式报告

- `run_robotwin2_five_body_paired_success_v1.py`
- `run_robotwin2_five_body_nested_n4_n8_paired_success_v1.py`
- `run_robotwin2_five_body_postformal_candidate_pool_v1.py`
- `evaluate_robotwin2_cross_embodiment_paired_success_v1.py`
- `evaluate_robotwin2_five_body_lobo_n1_n4_n8_oracle_v1.py`
- `materialize_robotwin2_nested_n1_n4_n8_final_report_v1.py`
- `run_robotwin2_five_body_lobo_offline_ablation_v1.py`

### 预注册、物化与验证

- `preregister_robotwin2_move_can_pot_five_body_lobo_v1.py`
- `materialize_robotwin2_stable_seed_roster_v1.py`
- `materialize_robotwin2_scripted_expert_root_supplement_binding_v1.py`
- `verify_robotwin2_move_can_pot_public_materialization_v1.py`

这一组文件不要在当前远程实验完成前改名、移动或合并。很多 watcher 会校验文件哈希；本地整理不应改变正在运行的远端只读代码目录。

## 2. 其他 VLA 代码如何分类

### OpenVLA / OpenVLA-OFT

文件名包含 `openvla` 的脚本属于 OpenVLA 历史与可复用接入线，主要包括：

- rollout / candidate branch 采集；
- action-Q、event world model 和 structured head 训练；
- OOF、sealed confirmation、校准和诊断；
- 外置 event critic plugin 示例。

当前五本体 v13 主线不直接调用这些文件；它们保留用于 OpenVLA 复现和未来多策略适配。入口优先看：

- `openvla_etsf_event_critic_plugin.py`
- `openvla_etsf_event_world_model.py`
- `train_openvla_etsf_event_world_model.py`
- `run_openvla_etsf_action_rerank.py`
- `launch_etsf_post_openvla_transfer.py`

### SmolVLA / Piper schema5/schema6

文件名包含 `smolvla` 或 `piper` 的脚本大多是前一阶段的单本体数据、schema5/schema6、paired-success 和 autonomous watcher 记录。它们不是当前五本体 LOBO 结果，但其中的 actor runtime、schema migration 和部署不确定性实现仍有复用价值。

注意：`robotwin2_smolvla_shared_event_critic_adapters_v1.py` 虽含 `smolvla`，它属于当前正式 v13 runtime，不能归档或删除。

### Stage0–Stage3 / 早期跨本体机制实验

`run_stage0_experiment.py`、`run_stage1.py`、`run_stage2.py`、`run_stage3.py` 及相邻的 `multibody` / `transfer` 文件保存早期事件—时钟解耦机制证据。它们不控制当前远端队列，但仍是论文机制来源和回归参考。

### 通用插件与适配器

不绑定具体 VLA 的可复用模块包括：

- `actor_agnostic_structured_event_time_plugin.py`
- `shared_event_critic_plugin_protocol_v1.py`
- `etsf_transfer_adapters.py`
- `etsf_policy_feature_action_bridge.py`
- `etsf_schema6_pose_quality.py`
- `etsf_torch_weights_only_compat_v1.py`

## 3. 文件状态规则

| 状态 | 处理规则 |
| --- | --- |
| current production | 保留原路径；变更后跑聚焦回归并更新交接文档 |
| reusable adapter/baseline | 保留；由本索引指向，不为“看起来整齐”而移动 |
| historical reproducibility | 不作为当前入口；Git 历史和文档仍需要时保留 |
| one-off probe / generated cache | 不进入仓库；放 `/tmp/etsf-*`，任务结束删除 |
| output / checkpoint / rollout / log | 只在运行服务器保存，禁止提交 Git |

新脚本命名建议：

```text
<verb>_robotwin2_<scope>_<role>_vN.py       # 当前五本体协议
<verb>_openvla_<scope>_vN.py                # OpenVLA 线
<verb>_smolvla_<scope>_vN.py                # SmolVLA 线
```

一次性检查不要再创建 `scripts/*_probe.py`。优先使用：

```bash
tmp_dir="$(mktemp -d /tmp/etsf-check.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -p no:cacheprovider <focused-tests>
```

`tmp_dir` 只用于一次性输入/输出；不要把 checkpoint 或正式结果放进去。
