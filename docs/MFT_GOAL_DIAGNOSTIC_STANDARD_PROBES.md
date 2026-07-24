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

Submission is pinned to `http://127.0.0.1:8002`. Port 8000 and the prior live
`a58` deployment are not accepted. `submit-standard` requires a sealed
`slurm-scheduler-prune-protection-cutover-receipt-v1` proving:

- commit `190f10de7e109a410fa03f743c24f2d682279ce8`;
- tree `810993c84d83b5632bdf48238a0df18579c3fa84`;
- release-manifest SHA
  `9d1fe324f4e11fee99aca5048d2355997b326c43919a0aeda38f8016b6ec573d`;
- live-launcher SHA
  `832847c134c002f7d1a80e2773297fc890f190ecc27d668f80e52f6f37015978`;
- verified `slurm-scheduler-prune-protection-v1` marker semantics;
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

Plans are independent and may be submitted in parallel up to the Scheduler's
live project and license admission limits. Collection is GET-only:

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
