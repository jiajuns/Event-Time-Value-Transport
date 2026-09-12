# V8 final archive on 5090

5090 root:

`/home/ps/EKSF/experiments/current/experiment2_ablation_fixed_20260901`

Final tables and audit:

- `results/final_v8/ablation_summary.csv`
- `results/final_v8/ablation_raw_metrics.csv`
- `results/final_v8/horizontal_with_same_abc_summary.csv`
- `results/final_v8/evaluation_manifest.json`
- `results/final_v8/integrity_and_cfc_use_audit.json`

Complete plot-ready training package:

- root: `results/plot_ready_v8`
- guide: `results/plot_ready_v8/README.md`
- all tables: `results/plot_ready_v8/tables`
- PNG and vector PDF figures: `results/plot_ready_v8/figures`
- generated-file hashes: `results/plot_ready_v8/MANIFEST_SHA256.txt`

The plot-ready package contains all formal and source-LOBO loss traces, every
formal step's parameter-group update norm, initial/final per-tensor and
per-group parameter differences, frozen per-seed/final metrics, checkpoint
inventory, and 18 rendered figure files (9 PNG + 9 PDF).

Published A+B+C checkpoints:

- `results/train_v8/e2_cfc_event_fixed/seed20260901/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260902/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260903/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260904/formal/step_3000.pt`
- `results/train_v8/e2_cfc_event_fixed/seed20260905/formal/step_3000.pt`

Checkpoint SHA-256:

- seed01 `90f8e7263bed20c4c390ea9270d1dbbca63a10d4e8eaf11f58aa902615397124`
- seed02 `3f587b4efb26950251f15f50848ec1e32b22317b7856214a387cbb77693ef5eb`
- seed03 `d7796de76e7cf5dc76fa5a052295092112469b114cffbb54e00cbc671ff05735`
- seed04 `d9f4516dea800f247060700d43199b45a0c94878157c02d81da3f9788bcd9ee6`
- seed05 `f1390692c239a9cf6d467b51b8eede6567842a307d30ee427736a74b9190ea58`

Archive verification performed on 2026-09-01:

- A: five runs, 3001 formal snapshots per run.
- A+B: five runs, 3001 formal snapshots per run.
- A+B+C v8: five runs, 3001 formal snapshots per run.
- All 15 final checkpoint hashes match HPC3 exactly.
- Same A+B+C checkpoints are referenced by the ablation and horizontal tables.
