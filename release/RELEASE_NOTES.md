# ETSF ICRA Artifact Release (V8)

本版本对应论文 **Cross-Embodiment Value Transport via Event Semantics, Reachability, and Execution Time** 的正式 artifact 主线。

包含：

- V8 五个 paired seeds 的仿真 event-successor critic checkpoint；
- 论文表格所用的最终指标、时间/CfC 审计、loss 与绘图数据；
- UMI 腕部 RGB-CfC 观察器，以及蔬菜、水果、锅盖三个任务的五折 value adapters；
- 三个从纯 SmolVLA checkpoint 追加 10,000 步 CfC-AWR 后训练得到的三相机策略；
- 对应的数据转换、preflight、训练、验收与可复现性脚本。

策略部署只加载 Release 中完整的 `pretrained_model` 目录，不加载 UMI、CfC、value critic、事件标签或权重 sidecar。三个策略的 prompt、相机与 state/action ABI 请严格按照 `docs/RELEASE_ASSETS.md`。

本版本删除了历史探索脚本、一次性 probes、测试目录、缓存以及未进入论文主线的配置。Event V4 策略 checkpoint 不在本 Release 中。
