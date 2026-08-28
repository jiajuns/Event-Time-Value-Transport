# Schema6 development300 身份解析与采集预注册

本文描述独立的 development300 v1 协议。它不修改、替换或声称兼容已冻结的 target v2。该协议只完成身份解析、身份冻结和采集命令预注册；它不授予采集执行能力，也不生成或授权 evaluation400。

## 输入边界

第一阶段只接受以下四类静态输入，并同时绑定逻辑 SHA 与文件 SHA：

- `etsf_smolvla_piper_schema6_target_development300_preregistration_v1`，精确包含 300 个按顺序预注册的 requested seed，以及固定的 train80 / internal30 / formal190 分区；
- full-horizon runtime contract v2b，必须是 200 step、离线模型加载、GPU 0，且不授权 test/evaluation，也不接受 fresh/confirmation 输入；
- 对完整 300 个 requested seed 候选池生成的 `strict private identity-set v1` 不相交证明；
- 一个按文件 SHA 冻结、只构造环境和执行 reset 的 adapter。该 adapter 不得导入或调用策略，不得 step，不得读取 reward、success、event、outcome、trajectory 或 label。

私有证明只公开 identity count 的承诺关系、集合 SHA、`intersection_count=0` 和签名；heldout identity 本身不进入本协议的任何公开文件。

## 三阶段状态机

### 1. 物化 reset-only authority

```bash
python scripts/materialize_smolvla_piper_schema6_development300_identity_authority.py \
  reset-authority \
  --preregistration DEVELOPMENT300_PREREG.json \
  --preregistration-file-sha256 PREREG_FILE_SHA \
  --preregistration-sha256 PREREG_LOGICAL_SHA \
  --runtime-contract RUNTIME_V2B.json \
  --runtime-contract-file-sha256 RUNTIME_FILE_SHA \
  --runtime-contract-sha256 RUNTIME_LOGICAL_SHA \
  --candidate-disjoint-attestation CANDIDATE_ATTESTATION.json \
  --candidate-attestation-file-sha256 CANDIDATE_ATTESTATION_FILE_SHA \
  --reset-adapter RESET_ONLY_ADAPTER.py \
  --reset-adapter-file-sha256 RESET_ADAPTER_FILE_SHA \
  --output development300_reset_authority.json
```

这个 authority 的能力上限是 300 次环境 reset。它明确禁止 environment step、policy import/forward、结果字段读取、采集以及 evaluation400 身份读取或执行。

### 2. 按预注册顺序解析身份

```bash
python scripts/materialize_smolvla_piper_schema6_development300_identity_authority.py \
  resolve-identities \
  --authority development300_reset_authority.json \
  --authority-file-sha256 RESET_AUTHORITY_FILE_SHA \
  --output development300_identity_resolution_receipt.json
```

解析器对 300 个 requested seed 各调用一次 reset，顺序必须与 preregistration 完全一致。稳定 setup 必须满足 `resolved_seed == requested_seed`；公开 receipt 只保留 scene、measured joints 和 commanded drive target 的 SHA，不保留原始值。

如果某个 setup 不稳定，adapter 只能返回 requested seed 和 `unstable` 状态。解析器记录该位置后继续处理下一个预注册 seed，不重试、不替换、不移动 split，也不依据结果挑选其他 seed。因为候选池恰好是固定的 300 个身份，任何一次 unstable 都会得到 incomplete receipt，并永久阻止该轮 collection identity authority。要继续只能在新的、显式版本化协议下重新预注册，不能静默补 seed。

解析完成时 receipt 必须证明：300 reset、0 step、0 policy query、0 outcome/label field read、0 HDF open。只有 300 个 setup 全部稳定、requested/resolved 各自全局唯一且逐项相等，才进入下一阶段。

### 3. 二次证明并冻结采集预注册

独立私有 attestor 必须对完整 selected requested/resolved identity set 再做一次不相交证明，并复用候选池证明中的同一个 heldout commitment。然后执行：

```bash
python scripts/materialize_smolvla_piper_schema6_development300_identity_authority.py \
  freeze-collection \
  --preregistration DEVELOPMENT300_PREREG.json \
  --preregistration-file-sha256 PREREG_FILE_SHA \
  --reset-authority development300_reset_authority.json \
  --reset-authority-file-sha256 RESET_AUTHORITY_FILE_SHA \
  --identity-receipt development300_identity_resolution_receipt.json \
  --identity-receipt-file-sha256 IDENTITY_RECEIPT_FILE_SHA \
  --selected-disjoint-attestation SELECTED_ATTESTATION.json \
  --selected-attestation-file-sha256 SELECTED_ATTESTATION_FILE_SHA \
  --future-collection-root /ABSENT/FUTURE/SCHEMA6/DEVELOPMENT300 \
  --output-directory /ABSENT/FUTURE/FROZEN/DEVELOPMENT300
```

`future-collection-root` 和 `output-directory` 在调用前都必须不存在。成功后输出目录和两个 JSON 文件变为只读：

- `collection_identity_authority.json`：冻结精确 80/30/190 身份和两次 attestation 的承诺链；
- `collection_preregistration.json`：按原始 300 requested seed 顺序列出 300 条 group command，每条固定 candidate index `[0,1,2,3]`，共 1200 个计划分支。

命令中的 HDF 路径只是未来输出位置，物化阶段不会创建或打开 HDF。所有命令都带有 `execution_authorized_by_preregistration=false`；实际采集仍需要一个独立、同时绑定 identity authority、runtime、collector 和输出根的 runner authority。

## Fail-closed 条件

任一条件都会阻止冻结或命令物化：输入文件 SHA 或逻辑 SHA 改变、runtime 不是 200-step v2b、resolver/adapter 实现 SHA 改变、attestation 目标集合或 heldout commitment 不匹配、requested 顺序改变、resolved 与 requested 不同、任一身份重复、split 不是精确 80/30/190、出现结果字段、receipt 不是 300 reset / 0 step / 0 policy / 0 outcome，或未来目录已经存在。

该链条不打开 formal190 的标签，不训练 adapter，不做 calibration，也不接触 evaluation400。formal190 的标签授权、采集 runner、训练与后续评估必须分别由新的、显式绑定的 authority 提供。

## 本地验证范围

单元测试只构造临时的合成 runtime 文件、合成 scene/joint reset 返回值和合成私有证明摘要。测试不连接远端，不读取真实 fresh、confirmation、formal validation、evaluation、trajectory、label 或 HDF 数据。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/jj/miniconda3/envs/spatial-lite-local/bin/python -m pytest -q \
  tests/test_materialize_smolvla_piper_schema6_development300_identity_authority.py
```
