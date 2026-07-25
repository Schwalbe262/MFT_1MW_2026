# Diagnostic actual-truth promotion to bounded Full FEA

`tools/mft_goal_truth_promotion.py` is a separate, explicit promotion path.
It does not change the diagnostic Standard CLI, and it does not treat a
diagnostic collection as a production Standard handoff.

The deadline campaign uses the v2 exact-cohort path. It binds exactly 24
sealed Standard submission receipts before promotion, then requires exactly
one authenticated collection for every bound Scheduler task. Every
collection, including a physically failing collection, is reauthenticated
through `mft_goal_diagnostic_standard_probe.authenticate_collection()`.
Dimensions, resonance, every active body/probe temperature, loss,
solver/library provenance, and the fixed cooling/operating identity are
recomputed from the actual diagnostic result. One missing, duplicate,
unexpected, tampered, or mixed-provenance collection aborts the whole
promotion.

A collection from a reviewed timeout retry or Scheduler
operational-pressure retry may replace its exact original logical slot. The
v2 loader follows the retry's sealed plan/submission/task ancestry back to
the inventory entry and accepts exactly one execution for that slot. It
rejects original plus retry, timeout plus pressure retry, duplicate retries,
mixed ancestry fields, and retries outside the frozen cohort.

Operational-pressure lineage additionally authenticates the plan's frozen
campaign claim-root reference and the submission's durable finalized claim.
The claim's logical authority task ID must route to the same cohort entry,
and its task ID must equal the collected execution. A copied/reparsed claim
root, a different immediate timeout ancestry, or a second direct/compound
winner for the same logical slot fails before truth classification.

Only the reauthenticated feasible observations enter the combined
non-dominated sort. The v2 manifest still seals a 24-row classification
ledger for both included and excluded observations. A surrogate prediction
or relaxed surrogate constraint cannot authorize promotion.

The legacy v1 path remains readable and accepts one to twelve passing
collections without a cohort inventory. It cannot prove completeness for the
24-task deadline cohort and must not be used for that result.

## 1. Seal the exact 24-task cohort

Create the inventory from the 24 immutable Standard submission receipts.
This command performs no Scheduler request or mutation.

```powershell
$Python = "C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe"
$Submissions = Get-ChildItem C:\evidence\standard-submissions\*.json |
  Sort-Object FullName

$Args = @("tools\mft_goal_truth_promotion.py", "create-cohort")
foreach ($Submission in $Submissions) {
  $Args += @("--standard-submission", $Submission.FullName)
}
$Args += @(
  "--output",
  "C:\evidence\standard-cohort-inventory.json"
)
& $Python @Args
```

`create-cohort` rejects any count other than 24, duplicate submission paths
or task IDs, invalid plan/submission lineage, stale search authority, or any
solver/library mixture.

## 2. Build the combined actual-truth Pareto result

```powershell
$Collections = Get-ChildItem C:\evidence\standard-collections\*.json |
  Sort-Object FullName

$Args = @(
  "tools\mft_goal_truth_promotion.py",
  "promote",
  "--cohort-inventory",
  "C:\evidence\standard-cohort-inventory.json"
)
foreach ($Collection in $Collections) {
  $Args += @("--standard-collection", $Collection.FullName)
}
$Args += @("--output", "C:\evidence\truth-promotion")
& $Python @Args
```

The output contains:

- `truth_validated_pareto_front.csv`
- `truth_pareto_manifest.json`

The v2 manifest seals the exact cohort inventory, all 24 source collection
identities, pass/fail reasons and actual constraint evidence. Feasible
candidates are deduplicated only when repeated actual observations agree.
Mixed pass/fail or differing actual truth for the same candidate aborts.
All remaining actual `(volume_L, total_loss_W)` observations are sorted
together using non-dominated sorting.

If all 24 observations fail, promotion still emits an auditable manifest and
a header-only Pareto CSV with `rank0_count=0`,
`full_plan_eligible=false`, and `zero_feasible_audited=true`. `plan-full`
then fails closed without creating a plan or submitting work.

## 3. Create rank-0 Full plans

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

## 4. Explicitly submit each Full plan

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

## 5. GET-only Full collection

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

## 6. Package both models and both result trees

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
& $Python tools\mft_goal_truth_promotion.py validate-cohort `
  --cohort-inventory C:\evidence\standard-cohort-inventory.json

& $Python tools\mft_goal_truth_promotion.py validate-truth `
  --truth-manifest C:\evidence\truth-promotion\truth_pareto_manifest.json

& $Python tools\mft_goal_truth_promotion.py validate-full-plan-set `
  --plan-set C:\evidence\truth-full-plans\full_plan_set.json

& $Python tools\mft_goal_truth_promotion.py validate-full-collection `
  --full-collection C:\evidence\<candidate>-full-collection.json

& $Python tools\mft_goal_truth_promotion.py validate-package `
  --manifest C:\evidence\<candidate>-truth-package\package_manifest.json
```
