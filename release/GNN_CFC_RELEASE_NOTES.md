# GNN+CfC experimental extension

This pre-release is attached to the `gnn-cfc-event-world-model` branch. The ICRA V8 implementation and its three-task CfC-AWR releases remain on `main`; this extension is **not** a replacement for those results.

The branch contains two related experimental paths:

- UMI relational event observers: a role-graph GNN followed by CfC, with V3 and V4 trained checkpoints and their selection/evaluation receipts. V3 was trained within a 4,000-step budget (selected at step 2,200). V4 was trained within a 4,000-step budget (selected at step 2,600), but **failed its frozen promotion gate** and remains experimental. In particular, the outside-tray false-success regression remains unresolved; do not use V4 as a calibrated RL reward.
- RoboTwin universal action-conditioned event world model: typed scene-graph GNN + CfC + task language/goal graph + canonical 14-D dual-end-effector action chunks. The included checkpoint is only a **10-step single-task smoke test** (532 train / 68 validation episodes from `move_can_pot`). It is not a completed multi-task pretraining model and is not validated for cross-embodiment deployment or SmolVLA action ranking.

The 24-trajectory small simulation matrix and its audit receipts are in the branch, but raw UMI videos and RoboTwin HDF5 are not redistributed. See [`docs/GNN_CFC_BRANCH.md`](https://github.com/jiajuns/Event-Time-Value-Transport/blob/gnn-cfc-event-world-model/docs/GNN_CFC_BRANCH.md) for code paths, training interfaces, caveats, and checkpoint contents.

Checkpoint archive: `etsf-gnn-cfc-experimental-checkpoints.tar.zst` (3,529,966 bytes; SHA256 `0c9e4453a5ee9310f74af8899235a343b87a1e1f385a0ef608806cd4a150911a`).

The archive contains three checkpoint directories—`umi_v3`, `umi_v4`, `robotwin_10step_smoke`—with receipts. No policy checkpoint or third-party YOLO weight is included.
