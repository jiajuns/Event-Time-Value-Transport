# TABLE V / horizontal shared A+B+C repair run

This isolated experiment preserves the original 600-rollout protocol while
repairing the progressive ablation implementation.  It never reads the formal
400-group branch-decision corpus.

The formal models all execute exactly 3000 optimizer steps.  A+B and A+B+C
use event-model pretraining for steps 1--2000 and matched ranking fine-tuning
for steps 2001--3000.  Every formal step checkpoint is retained.  The same
A+B+C step-3000 files are written into both the ablation and horizontal result
tables.

The optimized C stage has no zero gate or A+B fallback: every A+B+C event value
combines bounded GRU evidence with a required CfC time-evidence term.  The CfC
also carries independently supervised next-boundary state transport, elapsed
physical time, next-duration, temporal MC, and temporal ranking losses.
Its value contribution is scaled by a closed-form exposure derived from real
event dt, so a zero-time input has exactly zero temporal correction.
The physical reference is fixed at 0.5 per second (the source boundary median
is about 2 s); it is not learnable.  Mean interval exposure avoids cumulative
saturation.  Event-by-outcome temporal alignment prevents the CfC state from
becoming a source-body clock shortcut.
The CfC input includes both the canonical event-boundary state and the clipped
finite-difference state rate `(x_t-x_{t-1})/dt_eff`.  The same beta-scaled,
fixed-reference `dt_eff` drives the CfC closed-form update, so clock calibration
changes both the physical derivative and the continuous transition coherently.
Its success head stays on the GRU trajectory path, while event value adds the
ordered causal-chain prior.  This preserves within-event discrimination and
prevents target-body clock residuals from reversing event progress.
Success calibration uses the source-LOBO best-proper-score+1SE guard and only
then minimizes MAE; the target test labels remain untouched.

Hard startup gates:

- rollout relative-path/size manifest SHA-256:
  `a71666e8297c1b3919c2ec76de3e6b4e2a022adba0889d861f7435895b89d971`
- frozen source z-score SHA-256:
  `f84434f276e8a4f118e76b35c061e72af04a662a58df613503ebfb04041f5d14`

HPC3 entry point:

```bash
bash /data/user/leviccdong/EKSF/experiments/current/experiment2_ablation_fixed_20260901/hpc3/queue_fixed_v2.sh
```

The queue entry point deliberately reuses the already audited A and A+B v5
checkpoints and retrains only the changed A+B+C v8 model.  It refuses to start
unless all ten A/A+B step-3000 checkpoints exist.  Each new A+B+C seed must pass
its source-only real-dt-versus-zero-dt AUC gate before the target evaluation and
the final 15-run integrity audit are submitted.
