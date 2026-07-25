# Diagnostic Standard FEA truth probes

`tools/mft_goal_diagnostic_standard_probe.py` is a diagnostic-only path for
measuring the remaining Llt-surrogate uncertainty near the 2026-07-26 MFT
goal. It is separate from the production Standard-to-Full handoff. It cannot
submit a Full solve, create a production package, promote a candidate, or
change the Scheduler project.

Every artifact carries these fail-closed flags:

- `diagnostic_only=true`
- `standard_only=true`
- `production_eligible=false`
- `automatic_promotion=false`
- `full_submission_allowed=false`
- `production_package_allowed=false`

## Candidate contract

Selection authenticates the original 512-seed bundle ledger, every supplied
sealed terminal `result.json`, its terminal candidate CSV, and the pinned
`Llt_phys` model and metadata. A search-only bundle is valid diagnostic input;
this exception is opt-in and does not change the production handoff default.

A row is eligible only when:

- decoder, surrogate physical validity, and surrogate physicality all pass;
- every physical hard constraint except `Llt_robust_band` and
  `Llt_ensemble_disagreement` is non-positive;
- at least one of those two Llt uncertainty constraints is positive;
- the mean Llt is in 26.95–28.05 uH;
- a fresh, batched replay of the exact sealed `Llt_phys` model reproduces both
  Llt constraint values to the sealed numeric tolerance;
- dimensions, dynamic core-group constraints, and the fixed cooling/TIM/pad
  identity pass.

The default selection is eight candidates and the hard maximum is twelve.
Selection takes one candidate per available N1 stratum, then applies
deterministic farthest-point diversity over turns, core grouping, `cw1`,
dimensions, volume, and loss. It does not depend on a `least_violation` file.
It accepts either a final authenticated aggregate or arbitrary authenticated
terminal result files.

## Fixed Standard physics

The only profile is
`regression_260707/verify/profiles/goal_diagnostic_standard.json`. It is an
eight-core, 32 GiB, four-hour, eighth-symmetry Standard solve. The production
boundary remains unchanged: 1.5 m/s dual-fan cooling, 2 mm core-plate and WCP
pads, `k_ins=0.2`, and TIM conductivity 0.2 W/mK. The model is retained as
`symmetric.aedt` together with the complete adjacent
`symmetric.aedtresults` tree, a per-file content manifest, and
`.slurm-scheduler-preserve.json`.

Each plan binds all retained files to one run root and authenticates the
preservation-marker contract. Collection verifies the marker, AEDT SHA,
results-tree SHA, file count, byte count, Scheduler task status, exit code,
allocation/node identity, solver and library revisions, actual volume,
dimensions, loss components, physical Llt, resonance, every reported goal
temperature, and fixed-boundary readback.

## Scheduler cutover and admission gate

Submission is pinned to `http://127.0.0.1:8002`. Port 8000 and prior
deployments are not accepted. `submit-standard` requires a sealed
`slurm-scheduler-prune-protection-cutover-receipt-v1` proving:

- commit `0800a8d204da3cf772c4848814008b5755c92739`;
- tree `7377e526e81af6d171c136e9cc7346d9f96e140f`;
- release-manifest SHA
  `2b666a8fc6552cf2c18021d9a39fe6db1e3b76579bfd877852b5176d50e2a2c8`;
- live-launcher SHA
  `e26c3eeb0453cd5f049e9f3280213b46e9c194d149e139b34dc5d9947c137c3d`;
- verified `slurm-scheduler-prune-protection-v1` marker semantics;
- split attached-task CPU identity: the real bursty step envelope remains
  separately attested while the solver sees its reviewed task CPU contract;
- zero goal-active and zero all-nonterminal tasks at cutover.

Immediately before POST, the tool re-hashes the live launcher twice, reads
`/api/health` and `/api/licenses`, and requires a healthy Scheduler, enabled
and fresh license admission, no `blocked_reason`, and sufficient headroom for
the exact `MFT_1MW_2026v1` persistent license cost. The snapshots and hashes
are sealed into the submission receipt. Scheduler admission remains the
authoritative concurrent headroom reservation.

## Commands

Use the PyAEDT environment because the sealed ensemble includes LightGBM:

```powershell
$Python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$Root = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726'
$Bundle = "$Root\campaign_rolling512_a0977dd\bundle_manifest.json"
$Generation = "$Root\g0b_25target_6151\registry\generations\20260724T210635-800c21a0"
$Results = Get-ChildItem "$Root\slurm_offload\mft-goal-763dbb46ae74e1722969d2c2\harvest\task-*\result.json"
$ResultArgs = $Results | ForEach-Object { @('--source-result', $_.FullName) }
& $Python tools\mft_goal_diagnostic_standard_probe.py select `
  --bundle-manifest $Bundle `
  --generation $Generation `
  @ResultArgs `
  --limit 8 `
  --output "$Root\diagnostic-standard\selection"
```

After the final aggregate exists, prefer its all-seed authority:

```powershell
& $Python tools\mft_goal_diagnostic_standard_probe.py select `
  --bundle-manifest $Bundle `
  --generation $Generation `
  --aggregate-manifest "$Root\final-aggregate\aggregate_manifest.json" `
  --limit 8 `
  --output "$Root\diagnostic-standard\final-selection"
```

Create one independent plan per selected geometry. The solver revision must be
the pushed revision containing this tool and the diagnostic profile.

```powershell
$SolverRevision = (git rev-parse HEAD).Trim()
$LibraryRevision = 'e6b9b9d20a832ff5c3f7ca97218737a0b8650781'
& $Python tools\mft_goal_diagnostic_standard_probe.py plan `
  --selection-manifest "$Root\diagnostic-standard\selection\selection_manifest.json" `
  --candidate-physics-sha256 '<64-hex geometry SHA>' `
  --solver-revision $SolverRevision `
  --library-revision $LibraryRevision `
  --output "$Root\diagnostic-standard\plans\<geometry-prefix>"
```

Do not submit until the separate Scheduler cutover procedure has produced the
required receipt:

```powershell
& $Python tools\mft_goal_diagnostic_standard_probe.py submit-standard `
  --plan "$Root\diagnostic-standard\plans\<geometry-prefix>\diagnostic_plan.json" `
  --scheduler-cutover-receipt '<absolute cutover receipt path>' `
  --output "$Root\diagnostic-standard\submissions\<geometry-prefix>.json"
```

### Exact Scheduler operational-pressure retry

`plan-operational-pressure-retry` is separate from timeout retry semantics. It
accepts an original only when all of the following agree:

- task API: terminal `failed/failed`, null exit code, exact message
  `memory pressure hard limit after 3 attempts`, original 8 CPU/32 GiB/4 h
  resources, task/name/dedupe/project/allocation/job/account/node identity;
- public event API only: task GET before, the complete
  `/api/events?limit=1000` response, and task GET after. The two task
  readbacks must be identical. Event IDs must be unique and strictly
  descending, timestamps nonincreasing, and a full 1000-row window must
  reach the task's `created_at` (a shorter response is complete by API
  exhaustion). The complete normalized window, bounds, count, and SHA-256
  are sealed;
- within that full window, exactly one same-task/name/message `attempt 1/3`
  requeue, one `attempt 2/3` requeue, and one matching terminal failed
  cleanup are required, with
  `task.created_at <= attempt1 < attempt2 < final start <= finish < cleanup`.
  Requeue accounts must be nonempty. They are deliberately not required to
  equal the terminal account because a requeue may migrate allocations
  (the preserved live episode used three accounts); cleanup must equal the
  terminal task account. Any extra same-task `task_requeued` or
  `task_cleanup` event is rejected.

Missing attempt evidence, mixed timeout/pressure ancestry, a retry based on
another retry, or any change at the immediate pre-POST re-read blocks
submission. The retry retains identical fan 1.5 m/s, TIM/pads, operating
point, symmetry, and solver physics, but uses a distinct reviewed profile,
task/workdir, retained bundle and dedupe. A direct retry of the original
Standard task is 8 CPUs, 32 GiB, and 4 h. The one bounded compound form
`original failed/124 -> timeout retry failed by exact pressure evidence ->
pressure retry` is 8 CPUs, 32 GiB, and 8 h. No deeper chain is accepted.

Before any pressure plan is created, initialize the frozen campaign claim
root exactly once. This creates only local campaign authority files; it does
not call or mutate Scheduler and does not touch the separate Scheduler
project:

```powershell
$Python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$StrictCutover = 'C:\Users\peets\slurm_scheduler_runtime\deployment_candidates\41b3b9393684-strict-cpu-storage-admission-20260725\cutover_receipt.json'

& $Python -m tools.mft_goal_diagnostic_standard_probe `
  init-operational-pressure-claim-root
if ($LASTEXITCODE -ne 0) { throw 'claim-root initialization failed' }

& $Python -m tools.mft_goal_diagnostic_standard_probe `
  plan-operational-pressure-retry `
  --original-plan '<original diagnostic_plan.json>' `
  --original-submission '<original submission.json>' `
  --strict-node-name n116 `
  --output '<new pressure retry plan directory>'

& $Python -m tools.mft_goal_diagnostic_standard_probe `
  submit-operational-pressure-retry `
  --plan '<new plan directory>\diagnostic_operational_pressure_retry_plan.json' `
  --scheduler-cutover-receipt $StrictCutover `
  --priority 100 `
  --output '<new pressure retry submission.json>'
```

The plan seals a placement-invariant claim reference keyed by campaign,
candidate physics, logical authority task ID, and retry generation. An atomic
filesystem `mkdir` claim is acquired before Scheduler mutation. Only
`fresh_pending` authorizes the one submit call, and a distinct counter proves
the Scheduler client executed the post-claim guard exactly once immediately
before POST. `existing_pending` may recover only from exactly one matching
full public-API task; zero or multiple tasks fail closed and never re-POST.
`existing_finalized` also reauthenticates the one live sibling and never
re-POSTs. Direct/compound or strict/unplaced sibling plans for the same
logical slot collide at the same claim and cannot both win.

For multiple independent logical slots, create and submit one at a time.
After each receipt, GET the new task and live n116 capacity before proceeding.
Do not launch these commands in parallel: sequential admission preserves the
three currently available 8-CPU slots and gives every claim a durable API
readback. Optional same-allocation anchoring still requires all five
`--same-node-as-task-id` and `--expected-*` fields; partial/stale identity is
rejected. Collection is GET-only:

```powershell
& $Python tools\mft_goal_diagnostic_standard_probe.py collect `
  --plan "$Root\diagnostic-standard\plans\<geometry-prefix>\diagnostic_plan.json" `
  --submission "$Root\diagnostic-standard\submissions\<geometry-prefix>.json" `
  --scheduler-url http://127.0.0.1:8002 `
  --output "$Root\diagnostic-standard\collections\<geometry-prefix>.json"

& $Python tools\mft_goal_diagnostic_standard_probe.py validate-collection `
  --collection "$Root\diagnostic-standard\collections\<geometry-prefix>.json"
```

`authenticate_collection(path, predictor=None)` is the stable Python adapter
entry point. It returns exactly `schema_version`, `collection`, `plan`,
`params`, `selected`, and `submission`. A passing diagnostic result remains
diagnostic evidence only; a separate reviewed truth-promotion authority is
required before any Full solve.
