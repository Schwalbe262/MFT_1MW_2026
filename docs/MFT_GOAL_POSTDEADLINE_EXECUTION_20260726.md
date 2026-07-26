# MFT goal post-deadline execution record

Snapshot time: 2026-07-26 21:50 KST

## Status boundary

The original 2026-07-26 18:00 KST deadline was missed. None of the jobs in
this record can retroactively repair that deadline. They are diagnostic,
search-only, non-canonical runs until authenticated solver results are
collected and evaluated against the fixed goal contract.

The fixed physics boundary was not changed:

- dual fan, 1.5 m/s;
- insulation/TIM conductivity, 0.2 W/(m K);
- winding cooling-plate pad, 2 mm;
- core cooling-plate pad, 2 mm;
- exterior limits, 1200 mm x 1000 mm x 750 mm;
- resonance minimum, 15 kHz;
- winding maximum temperature, 100 degC;
- core maximum temperature, 120 degC.

## Why the Scheduler was empty

The period with no active goal job was an execution sequencing failure, not a
Slurm capacity shortage. Existing scientific tasks had already reached
terminal states:

- the provisional Full task 96307 timed out after 12 hours without a retained
  solver result;
- corrected-thermal task 96313 timed out before native mesh completion;
- the frozen Standard cohort had only timeout, cancellation, dependency, or
  memory-pressure terminals and no authenticated collection.

Work then remained focused for too long on checkpoint authentication,
fail-closed retry-plan construction, evidence sealing, and Web UI recovery.
Those checks were necessary, but a fresh independent Slurm retry should have
been prepared and submitted in parallel. The empty queue observed by the user
was therefore real, and the delay was not caused by the Scheduler refusing
available work.

## Symmetric-first finalization policy

Authenticated symmetric/Standard FEA is now the primary candidate-selection
gate. Candidate-by-candidate Standard-to-Full continuation is disabled. Full
task 96326 continues only as a diagnostic reference; after one symmetric
candidate is explicitly selected, at most that one candidate may receive a
final Full validation.

The current search is bounded to the existing NSGA-II solutions. The preferred
anchor order is official candidate 12, then 6, with 1 and 5 as backups. If any
authenticated symmetric result passes every fixed constraint, selection stops
and ranks passing results by actual minimum normalized constraint margin,
actual loss, and actual volume. Only when no current result passes and the
authenticated surrogate-to-FEA residual is small may a local symmetric batch
be prepared, capped at three neighbors per round and two rounds. The local
workflow has no Scheduler submission capability.

## Parallel recovery lanes

The effective candidate-selection pipeline has seven unique symmetric/Standard
lanes. Full task 96326 runs separately as a reference. Task 96329 remains
preserved in the queue, but task 96333 replaces it as the effective official
candidate 8 lane because n114 is in Slurm drain.

| Task | Lane | Candidate | Node | Resources | Scheduler envelope | Snapshot state |
|---:|---|---|---|---|---:|---|
| 96324 | corrected thermal symmetric | b7c30cb70b95 | n111 | 8 CPU, 294912 MB | 45000 s | FAILED, no temperature result |
| 96325 | Standard symmetric | efffb6518d4e | n107 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96326 | Full diagnostic reference | b7c30cb70b95 | n116 | 16 CPU, 98304 MB | 86400 s | RUNNING |
| 96327 | Standard symmetric hedge | b6a83bfc7212 | n109 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96328 | official 6 Standard symmetric | 2772aed82a8c | n113 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96329 | official 8 original lane | 622097dde126 | n114 | 8 CPU, 98304 MB | 45300 s | QUEUED, superseded for selection |
| 96330 | official 1 Standard symmetric | 896084a59793 | n110 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96331 | official 12 Standard symmetric | 828cb282cf4f | n112 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96332 | official 5 Standard symmetric | 909d249ebe45 | n115 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96333 | official 8 Standard failover | 622097dde126 | n111 | 8 CPU, 98304 MB | 45300 s | RUNNING |

At this snapshot, eight physical nodes have active allocations: seven
symmetric/Standard candidate lanes and the one Full reference. Strict
placement is satisfied for every running task. Task 96324 passed its mesh
preflight but Icepak failed before launching a valid Fluent solve, so it
produced no authenticated temperature residual and is excluded from local
correction rather than classifying the design as infeasible. The secondary
NaN JSON failure occurred while writing its failure receipt and is not the
scientific root cause.

## Pareto and scientific truth

The authenticated 512-seed aggregate contains 163840 terminal rows and 133563
unique geometries. Global non-dominated sorting was recomputed across all
authenticated terminal rows rather than merging seed-local fronts.

- physically feasible rows: 0;
- production-feasible Pareto rows: 0;
- audit objective front rows: 22.

Accordingly, the existing Pareto files are search evidence, not a passing
production Pareto front. No resonance or temperature PASS is claimed in this
record. Actual solver outputs from the running lanes must be collected,
authenticated, and re-ranked before any scientific or production claim.

## Durable evidence

- Aggregate manifest:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\aggregate_rolling512_d4e4d60\aggregate_manifest.json`
- Corrected-thermal task 96324 receipt:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_symmetric_r6\postdeadline_symmetric_r6_submission.json`
- Standard task 96325 evidence:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_efffb6518d4e_r1_n107_260726\artifacts`
- Standard task 96327 evidence:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_b6a83bfc7212_r2_n109_260726\artifacts`
- Full task 96326 sealed plan and single-POST receipt:
  `artifacts/mft_goal_postdeadline_full_retry_plan_v1.json`
- Effective seven-lane Standard post-success state:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_postsuccess_96325_96327_96328_96330_96331_96332_96333_v5\state.json`
- Official 8 n111 failover final seal:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_official8_failover_622097dde126_n111_260726_v1\submission\final_seal.json`
- Symmetric-first local trust waiting manifest:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\local_trust_waiting_20260726_v2\acquisition_manifest.json`
- Live UI:
  `http://127.0.0.1:8010/`

The Scheduler repository remains a separate project. No Scheduler source was
mixed into the MFT repository for these submissions.
