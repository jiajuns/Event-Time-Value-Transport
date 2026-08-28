# Source63 requested/resolved seed split recovery (2026-08-28)

## Incident

The first immutable source training run
`/home/user/etsf_smolvla_schema5_native_source_training_r7d_20260828`
failed closed before the first optimizer step.  Its frozen split is explicitly
`split_unit=requested_seed_logical_group`, while the counterfactual trainer
indexed every HDF group only by `resolved_seed`.  Six RoboTwin resets resolved
to a different identity, so those six logical groups could not be assigned.
No fresh/confirmation label was read and no target data was used.

The differing requested/resolved pairs were:

- 100100088 -> 100100089
- 100100097 -> 100100098
- 100100101 -> 100100103
- 100100105 -> 100100107
- 100100113 -> 100100114
- 100100139 -> 100100140

## Repair

`GroupDescriptor` now records both identities. `read_split_manifest()` selects
the lookup identity from the frozen `split_unit`: requested-seed manifests map
requested identities to resolved logical group keys; resolved-seed manifests
retain the prior behavior. The original 44/14/5 membership and label-blind
split decision are unchanged.

Regression test:
`test_requested_seed_split_maps_to_resolved_logical_groups`.
The counterfactual/source-launcher suite passed 45 tests.

## Immutable recovery deployment

- code root: `/home/user/etsf_smolvla_source63_training_code_r7e_20260828`
- repaired trainer SHA256:
  `d625e76e3658072006bd9315fe4baf92f1fc7f5408cd76ce27df77544b82dd55`
- implementation bundle SHA256:
  `a6c815b7f23373be45ac8f289ac9cd2a5b7db124b8fd178ccb2301e48dee35fe`
- output root:
  `/home/user/etsf_smolvla_schema5_native_source_training_r7e_20260828`
- source static plan SHA256:
  `7283aadee5b667b71cad0aac34d9d7e25237e40e4d206950ab6a8e850b3c436a`
- detached source watcher PID: `1926219`
- source detach receipt SHA256:
  `e5789f9aa96908eff8908ac41a75c12fbb98f3b0c93bea29f3c06e8f5dbf65b4`

The exact requested-to-resolved mapping was re-audited on the remote CPU
without opening label datasets: train=44, validation=14, sealed test=5.

Downstream recovery watchers were preregistered against the new source plan:

- LOBO root `/home/user/etsf_multibody_lobo_autonomous_r8c_20260828`, PID
  `1926829`, static plan
  `f006e8f172d09326eb98d0c234711cf04ce7987f8fc2d9fb40519a3baf75d4a2`.
- Schema6 root
  `/home/user/etsf_smolvla_piper_schema6_autonomous_r10_20260828`, PID
  `1927458`, static plan
  `ec59eb25fa2ac573f06d1ac7ae6c9bc2accc28bbf1140374309d375205775268`.

All three processes have PPID 1 and survive client shutdown. At deployment
time they were waiting for the unrelated OpenVLA LIBERO-goal evaluation to
release the designated RTX 4090.

The unrelated official evaluations are themselves launched sequentially by
`bash /home/user/openvla-repro/run_all_full.sh` (PID 1830377).  A short empty
GPU interval between suites is not an authorization boundary.  Detached guard
PID `1934347` therefore holds
`/tmp/etsf_smolvla_schema5_source63_gpu0.lock` until PID 1830377 exits; it then
releases the lock automatically.  This prevents source training from racing a
later official suite while preserving both workloads.
