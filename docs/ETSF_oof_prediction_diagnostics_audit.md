# ETSF OOF structured prediction diagnostics 独立审计

审计范围仅为代码、冻结 OOF manifest 合同和合成数据测试；没有读取正在采集的 development 标签、
任何 HDF5 数据或 fresh50。

## 审计结论

原诊断足以生成描述性 held-out 指标，但不足以单独证明“准确预测”。主要缺口是：success 缺少
PR-AUC 和组内 pair skill；事件/成败缺少 macro-F1；success 缺少绝对 ECE 上限；duration 没有
event×body 外折基线；AURC 没有随机排序参照；post-predicate 只有易被负类主导的 micro 指标，
没有逐谓词 skill 门；所有 head 也没有统一、与 reranking guard 解耦的 fail-closed 判定。此外 raw
structured block 未强制三成员、id 范围、mask 关系以及 terminal label 与候选 row 一致，存在形状
正确但语义错位仍被接受的风险。

以上缺口已经补齐。当前证据链为：

1. manifest 签名确定 100/250 个 logical groups 及唯一 owner fold；重复、遗漏或 fold 篡改拒绝；
2. 每个 raw row 固定三成员，candidate name/order、数值有限性和二元标签受检；
3. structured sample 固定三成员，event/outcome id、duration、predicate、body/policy、mask 均受检；
4. terminal samples 必须位于 structured block 开头，其 name 和 success label 必须逐项匹配 raw
   candidate row；
5. success temperature 对目标 fold 的校准只用另外四折；event prior、outcome prior 和
   event×body duration median 同样只用另外四折标签；
6. paired 差异按 logical group 聚类 bootstrap，避免候选数或长轨迹 transition 数较多的 group
   不成比例地缩小置信区间；
7. success ECE 固定上限为 0.10；每个 post-predicate 独立检查正负支持、macro-F1，以及相对
   other-fold smoothed prevalence 的 Brier 组级 CI，任一缺失均 fail closed；
8. deployment 前四候选进入主判定，第五训练候选只进入签名附录；
9. 独立 prediction adequacy 只产生 development predictive-skill 结论，对 reranking guard 和
   fresh50 没有任何权限。

## 仍然不能证明的事项

- 没有任务方预注册的 object position 容差、duration deadline 或 calibration cost，因此通过通用
  skill gate 仍不等于“绝对准确”或“满足控制安全要求”。
- 同一 logical group 内 transitions 高度相关；当前 CI 已在 group 层聚类，但只有一个任务/主体时
  不能外推跨任务、跨本体或跨策略。
- recovery 只有在 checkpoint 的训练合同为 `recovery_supervised=true` 且每类支持达到阈值时才进入
  adequacy；否则必须写不可评估。
- development OOF 不是 fresh confirmation，预测 skill 也不等于在线重排提高成功率。

因此，当前诊断在正式 OOF 产物生成后可以严谨回答“是否在当前 development 分布上显著优于冻结
baseline”，但对更强的“绝对准确、跨本体、提高在线成功率”结论仍必须分别提供任务阈值、迁移实验
和 guard-authorized fresh confirmation。
