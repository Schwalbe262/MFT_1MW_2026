# MFT goal post-deadline execution record

Snapshot time: 2026-07-26 18:13 KST

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

## Parallel recovery lanes

All four jobs below have no `same_node` dependency and use distinct task names,
dedupe keys, nodes, and retained-output paths.

| Task | Lane | Candidate | Node | Resources | Scheduler envelope | Snapshot state |
|---:|---|---|---|---|---:|---|
| 96324 | corrected thermal symmetric | b7c30cb70b95 | n111 | 8 CPU, 294912 MB | 45000 s | RUNNING |
| 96325 | Standard symmetric | efffb6518d4e | n107 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96326 | Full diagnostic retry | b7c30cb70b95 | n116 | 16 CPU, 98304 MB | 86400 s | RUNNING |
| 96327 | Standard symmetric hedge | b6a83bfc7212 | n109 | 8 CPU, 98304 MB | 45300 s | RUNNING |

At this snapshot, Scheduler placement contracts were satisfied on all four
nodes. Tasks 96325 and 96327 had entered native Maxwell solve stages. Task
96324 had opened the authenticated corrected-thermal project and entered the
fresh mesh path. Task 96326 had started on n116; its Full solver invocation is
guarded to 79200 seconds inside the 86400-second Scheduler envelope.

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
- Live UI:
  `http://127.0.0.1:8010/`

The Scheduler repository remains a separate project. No Scheduler source was
mixed into the MFT repository for these submissions.
