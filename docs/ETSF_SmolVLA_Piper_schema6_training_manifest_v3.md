# Schema6 development300 training manifest v3

`scripts/materialize_smolvla_piper_schema6_training_manifest_v3.py` is a new,
create-once bridge from the development300 collection terminal to the existing
Schema6 adapter trainer. It does not modify or reinterpret the historical v2
materializer.

## Accepted chain

The command requires explicit file and logical SHA256 bindings for all of:

- the successful
  `etsf_smolvla_piper_schema6_development300_collection_terminal_receipt_v1`;
- its `collection_runner_authority_v1`;
- the original signed
  `target_development300_preregistration_v1`;
- the signed `collection_identity_authority_v1`;
- the exact current `train_smolvla_piper_schema6_embodiment_adapter.py`.

The runner authority must still validate its own collection preregistration,
runtime-v2b, event specification, collector, runtime adapter, sealed worker,
runtime Python and support-code closure. The terminal collection root must be
fully read-only and contain no symlink or forbidden fresh/confirmation/test
path. The detached receipt, static plan, terminal state, empty staging area and
all 300 launch/stage/group/reset/accounting receipts are revalidated.

Each group must retain its preregistered global ordinal, split ordinal,
requested/resolved seed and pair id. Requested seeds, resolved seeds, pair ids,
split indices, HDF paths and derived logical group ids are independently unique;
the three sets are disjoint and cover exactly 80 adaptation-train, 30 internal
validation and 190 formal validation groups.

## HDF boundary

No HDF5 byte is opened or hashed. The HDF path is checked with `lstat`, must be
a non-symlink regular read-only file at the exact command path, and its recorded
SHA is accepted only through the terminal-bound stage and sealed-group receipt
chain. Registry, pose-spec and receipt JSON metadata are opened and validated;
formal HDF/trajectory/label contents remain sealed.

## Create-once outputs

The output directory must not exist. A successful run creates it once and
freezes these five JSON files read-only:

- `schema6_training_manifest_v3_compat.json`: trainer-compatible manifest v1;
- `schema6_target_partition_v3.json`: adaptation110/formal190 partition;
- `schema6_external_group_split_v3.json`: exact train80/validation30/test190;
- `schema6_expected_manifest_split_v3.json`: receipt consumed by the trainer;
- `schema6_training_manifest_v3_receipt.json`: complete upstream/output lineage.

The expected-receipt format is
`etsf_smolvla_piper_schema6_expected_manifest_split_v3` with
`split_profile=development300_v3`. It does not authorize opening formal190
labels for training/checkpoint selection and does not generate or authorize any
evaluation400 identity or command.

## Invocation fields

```text
--collection-root
--terminal-receipt
--expected-terminal-receipt-file-sha256
--expected-terminal-receipt-sha256
--runner-authority
--expected-runner-authority-file-sha256
--expected-runner-authority-sha256
--target-preregistration
--expected-target-preregistration-file-sha256
--expected-target-preregistration-sha256
--identity-authority
--expected-identity-authority-file-sha256
--expected-identity-authority-sha256
--bound-trainer
--expected-bound-trainer-file-sha256
--output-directory
```

The required upstream JSON fields are not optional projections: terminal must
contain the exact success-v1 schema including runner/plan SHA, exact counts,
gap-free order and stage-order SHA; identity authority must expose all 300
`selected_rows`; target preregistration must expose all 300 `groups`, its
partition and seed-generation base; runner authority must expose the signed
collection preregistration, implementation closure, event specification,
runtime-v2b and exact commands with absolute output paths.
