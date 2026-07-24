# Diagnostic actual-truth promotion to bounded Full FEA

`tools/mft_goal_truth_promotion.py` is a separate, explicit promotion path.
It does not change the diagnostic Standard CLI, and it does not treat a
diagnostic collection as a production Standard handoff.

The path accepts one to twelve sealed
`mft-goal-diagnostic-standard-collection-v1` files. Every input is
reauthenticated through
`mft_goal_diagnostic_standard_probe.authenticate_collection()`. Dimensions,
resonance, every active body/probe temperature, loss, solver/library
provenance, and the fixed cooling/operating identity are recomputed from the
actual diagnostic result. A surrogate prediction or relaxed surrogate
constraint cannot authorize promotion.

## 1. Build the combined actual-truth Pareto result

```powershell
$Python = "C:\Users\peets\anaconda3\python.exe"

& $Python tools\mft_goal_truth_promotion.py promote `
  --standard-collection C:\evidence\probe-01\collection.json `
  --standard-collection C:\evidence\probe-02\collection.json `
  --standard-collection C:\evidence\probe-03\collection.json `
  --output C:\evidence\truth-promotion
```

The output contains:

- `truth_validated_pareto_front.csv`
- `truth_pareto_manifest.json`

Candidates are deterministically deduplicated by candidate physics identity.
All remaining actual `(volume_L, total_loss_W)` observations are sorted
together using non-dominated sorting. The manifest seals the complete ranked
rows, source collection identities, CSV SHA-256, and no-surrogate authority
flags.

## 2. Create rank-0 Full plans

```powershell
& $Python tools\mft_goal_truth_promotion.py plan-full `
  --truth-manifest C:\evidence\truth-promotion\truth_pareto_manifest.json `
  --solver-revision <40-to-64-hex-solver-revision> `
  --library-revision <40-to-64-hex-library-revision> `
  --limit 3 `
  --output C:\evidence\truth-full-plans
```

Only rank-0 actual-truth rows can receive a plan. The default and hard maximum
are three. Plans use the exact diagnostic candidate parameters and the
reviewed Full profile:

- `full_model=1`
- `thermal_symmetry=full`
- 16 CPUs, 98,304 MB, 12-hour timeout
- fan velocity `1.5 m/s`, dual fans
- core-plate and WCP pads `2.0 mm`
- thermal pad conductivity `0.2 W/(m K)`

Creating plans never submits a Scheduler task.

## 3. Explicitly submit each Full plan

Submission requires the reviewed Scheduler cutover receipt, a fresh live
`/api/health` and `/api/licenses` admission result from the isolated Scheduler
at `http://127.0.0.1:8002`, and a fresh 16-core AEDT license snapshot.

```powershell
& $Python tools\mft_goal_truth_promotion.py submit-full `
  --plan C:\evidence\truth-full-plans\<candidate>\full_plan.json `
  --scheduler-cutover-receipt C:\evidence\scheduler-cutover.json `
  --license-snapshot C:\evidence\license-headroom.json `
  --output C:\evidence\<candidate>-full-submission.json
```

There is no automatic Full submission command in the diagnostic tool. A
failed cutover, launcher identity, health, license freshness/headroom, truth
rank, retained-artifact marker, or provenance check blocks Scheduler POST.
The Scheduler repository/project remains separate from the 1MW MFT
repository.

## 4. GET-only Full collection

After the task is terminal-success:

```powershell
& $Python tools\mft_goal_truth_promotion.py collect-full `
  --plan C:\evidence\truth-full-plans\<candidate>\full_plan.json `
  --submission C:\evidence\<candidate>-full-submission.json `
  --scheduler-url http://127.0.0.1:8002 `
  --output C:\evidence\<candidate>-full-collection.json
```

Collection performs Scheduler GETs only. It verifies the 16-CPU task
execution, strict result identity, `full_model=1`, full thermal symmetry,
runtime license evidence, the Full AEDT receipt/results-tree manifest/marker,
and reauthenticates the retained diagnostic symmetric evidence for the same
candidate.

## 5. Package both models and both result trees

Packaging is allowed only when both diagnostic actual truth and Full actual
truth pass every hard constraint.

```powershell
& $Python tools\mft_goal_truth_promotion.py package `
  --full-collection C:\evidence\<candidate>-full-collection.json `
  --scheduler-url http://127.0.0.1:8002 `
  --output C:\evidence\<candidate>-truth-package
```

The package contains and re-hashes:

- `symmetric_model.aedt`
- `symmetric_model.aedtresults/`
- `full_model.aedt`
- `full_model.aedtresults/`
- sealed source evidence copies
- `package_manifest.json`

The package manifest records local absolute paths, SHA-256 and byte size for
both AEDT files and every results-tree member, actual constraints and margins,
fixed cooling identity, task evidence, truth-manifest identity, and explicit
no-mixed-model/no-surrogate flags. Missing data, a failed Full result, a
different candidate, or any byte/tree-hash drift blocks packaging or later
validation.

## Read-only validation commands

```powershell
& $Python tools\mft_goal_truth_promotion.py validate-truth `
  --truth-manifest C:\evidence\truth-promotion\truth_pareto_manifest.json

& $Python tools\mft_goal_truth_promotion.py validate-full-plan-set `
  --plan-set C:\evidence\truth-full-plans\full_plan_set.json

& $Python tools\mft_goal_truth_promotion.py validate-full-collection `
  --full-collection C:\evidence\<candidate>-full-collection.json

& $Python tools\mft_goal_truth_promotion.py validate-package `
  --manifest C:\evidence\<candidate>-truth-package\package_manifest.json
```
