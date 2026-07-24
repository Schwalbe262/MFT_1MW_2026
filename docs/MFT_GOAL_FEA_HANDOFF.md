# MFT goal Pareto-to-FEA handoff

`tools/mft_goal_fea_handoff.py` is the fail-closed transition from the
authenticated multi-seed search result to retained Standard and Full AEDT
projects. It does not edit the separate Scheduler repository.

## Authority required for submission

Only `plan --aggregate-manifest ... --bundle-manifest ...` can create a
submission-eligible plan. The command reauthenticates every task in the sealed
bundle and every corresponding terminal seed result, requires at least 32
unique seeds with all four fixed-primary-turn strata (5, 6, 7, and 8), checks
the complete 320-member generation-300 terminal population, deduplicates
physical geometry, and recomputes the global non-dominated ranks.
Both Standard and Full repeat that complete source/bundle/seed authentication
and global NDS recomputation immediately before calling the Scheduler. A
jointly edited and re-sealed plan and selected-candidate file therefore cannot
be used as submission authority.

`--standard-candidates` plus an operator SHA-256 remains available only for
byte-exact inspection. Such a plan is permanently marked non-production and
both submit commands reject it.

The selected terminal row must retain its exact decoded geometry, goal
constraint vectors, task/result identities, fixed 1 MW operating point, and
fixed cooling boundary: dual 1.5 m/s airflow, 2 mm pads, 0.2 W/(m K) TIM and
insulation conductivity, and both cold plates enabled.

## Stage order and resources

The immutable profiles are:

- Standard: eighth symmetry, 8 CPUs, 32 GiB, four-hour timeout.
- Full: full geometry, 16 CPUs, 96 GiB, twelve-hour timeout.

Both use the standalone AEDT backend and `keep_project=1`. Full submission is
impossible until the sealed Standard collection passes every active winding
body/probe limit (100 C), every active core body/probe limit (120 C), and the
complete physical specification. The Full submission receipt permanently
binds the exact Standard task, result, collection, and PASS gate that preceded
it.

The 16-core command requires a fresh operator-side admission snapshot. Queue
time cannot stale the solve: after allocation and the Scheduler startup
stagger, `tools/mft_runtime_license_snapshot.py` runs on the compute node,
checks the exact license server and required feature headroom, and generates
the revision-bound core authorization immediately before solver startup.

## Retained projects

After a successful solve, the Scheduler client opt-in copies exactly one AEDT
project to a task-dedupe-specific directory beneath the task `remote_cwd`.
Each Standard and Full root contains:

- `symmetric.aedt` or `full.aedt`;
- a receipt binding artifact size/SHA-256, task dedupe, profile, solver and
  library revisions, and project identity;
- `.slurm-scheduler-preserve.json`, using the Scheduler
  `slurm-scheduler-prune-protection-v1` contract;
- text-safe base64 chunks used to transport binary AEDT bytes through the
  existing 1 MiB text remote-file API.

Do not remove either preservation marker or its remote root before `package`
finishes. Scheduler deployment must include prune-protection commit
`ad386bce8a8046dae531e3ed969dd6eadc7711ce`; this repository only writes and
verifies the marker.

## Command sequence

```text
plan
submit-standard
collect
gate
submit-full
collect
package
```

`collect` and `package` are GET-only and reject a Scheduler URL different from
the submission origin. `package` reconstructs and rehashes both binary AEDT
files, then emits the profiles, parameters, submissions, collections, gate,
result identities, Standard-to-Full transition, and both preservation-marker
payloads/hashes.

Operational assumptions:

- The solver and PyAEDT-library revisions are full 40-hex commits available to
  compute nodes. The solver revision must contain this handoff and runtime
  license helper.
- The selected aggregate, bundle, every bundle task, and every terminal seed
  result remain byte-identical and readable through both submit commands.
- Submission receipts, collections, and the Standard gate remain readable
  until `package` completes.
- No live task is submitted by repository tests.
