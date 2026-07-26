# MFT goal post-deadline execution record

Snapshot time: 2026-07-26 23:17 KST

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

The current search is bounded to the existing NSGA-II solutions. Under the
user-stated geometry, resonance, and temperature constraints, official
candidate 5 is the direct-finalization anchor because it has the best predicted
minimum constraint margin among the official Standard 12:

- candidate geometry SHA: `909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42`;
- exterior dimensions: 1194.38 mm x 961.40 mm x 707.00 mm;
- predicted half-magnetizing resonance: 15078.46 Hz;
- predicted winding maximum: 99.64 degC;
- predicted core maximum: 119.40 degC;
- predicted objective loss: 5461.28 W;
- predicted objective volume: 811.83 L.

These are surrogate predictions, not authenticated thermal PASS values.
Candidate 5 also remains a near-feasible search fallback in the broader
internal constraint set, so only the actual symmetric solve can promote it.
If any authenticated symmetric result passes every fixed constraint, selection
stops and ranks passing results by actual minimum normalized constraint margin,
actual loss, and actual volume. Only when no current result passes and the
authenticated surrogate-to-FEA residual is small may a local symmetric batch
be prepared, capped at three neighbors per round and two rounds. The local
workflow has no Scheduler submission capability.

## Parallel recovery lanes

The effective candidate-selection pipeline has seven unique symmetric/Standard
lanes. Full task 96326 runs separately as a reference. Task 96329 remains
preserved in the queue, but task 96333 replaces it as the effective official
candidate 8 lane because n114 is in Slurm drain. For candidate 5, corrected
same-node direct-Analyze task 96338 replaces task 96332 for selection while
the original remains running and preserved.

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
| 96332 | official 5 Standard symmetric | 909d249ebe45 | n115 | 8 CPU, 98304 MB | 45300 s | RUNNING, superseded for selection |
| 96333 | official 8 Standard failover | 622097dde126 | n111 | 8 CPU, 98304 MB | 45300 s | RUNNING |
| 96337 | official 5 direct-Analyze v1 | 909d249ebe45 | n115, same allocation as 96332 | 8 CPU, 98304 MB | 45300 s | FAILED before EM, core-auth digest drift |
| 96338 | official 5 direct-Analyze corrected retry | 909d249ebe45 | n115, same allocation as 96332 | 8 CPU, 98304 MB | 45300 s | RUNNING, selection effective |

At this snapshot, the candidate lanes continue in parallel and task 96338
shares the already active n115 allocation with task 96332. Strict placement is
satisfied for every running task. Tasks 96325, 96327, 96328, and 96338 have
each emitted the native `Solving design setup ThermalSetup` marker; none has
yet emitted an authenticated temperature result or terminal result. Task
96324 passed its mesh
preflight but Icepak failed before launching a valid Fluent solve, so it
produced no authenticated temperature residual and is excluded from local
correction rather than classifying the design as infeasible. The secondary
NaN JSON failure occurred while writing its failure receipt and is not the
scientific root cause.

## Direct-Analyze symmetric fast path

All seven effective Standard lanes completed their Maxwell/capacitance/loss
stages and built Icepak, but none had produced an authenticated temperature
result at this snapshot. They were blocked in or immediately around the
standalone native `GenerateMesh("ThermalSetup")` preflight. Task 96327 had
emitted the native `Solving design setup ThermalSetup` marker, but still had no
terminal or temperature result.

Revision `d4e78cfa757888f7c47964aa392e4ff96f1e453f` adds an explicit,
versioned, eighth-symmetry-only direct-Analyze path. It is disabled by default.
It is enabled only with the exact token
`MFT_SYMMETRY_THERMAL_DIRECT_ANALYZE=standard-eighth-direct-analyze-v1`.
The path verifies standalone/eighth eligibility, the fixed fan/TIM/pad
contract, native setup identity, TIM material readback, and native mesh
assignment readback. It omits only the separate `GenerateMesh` API call and
then uses the existing native Analyze, convergence, temperature-extraction,
and scientific truth gates. The focused regression set passed 145 tests.

Three same-allocation support tasks were completed on task 96332's n115
allocation without changing the running solver:

| Task | Purpose | Result |
|---:|---|---|
| 96334 | Read-only live project inventory | completed exit 0 |
| 96335 | Read-only AEDT-results inventory | completed exit 0; 980 MiB results tree |
| 96336 | Immutable reflink clone for continuation | completed exit 0; 991 MiB clone |

Task 96336 proved the source identity before and after copying and matched the
source and clone content-tree hash:
`7587cc8b895fd9aa1b37e1bc10ef11d7d26d6fcbca7dd313fb0769d6f5ce18bb`.
The clone is read-only at
`/enroot/mft_goal_direct_clone_t96332_909d249ebe45_v1`. A continuation must
copy it to a fresh writable directory, must not modify task 96332 or this
clone, and must remain on n115/allocation 14650.

A fresh direct-Analyze lane was instead attached to the same live
n115/allocation 14650 using 8 CPU and 98304 MB. Its first execution, task
96337, was an operational failure before EM: the command checked out solver
revision `d4e78cf` but inherited the revision-bound 8-core authentication
digest for `a1e4f70`. It performed no EM, thermal solve, or scientific
classification. The corrected retry task 96338 uses the exact digest
`0305537ff06f209faa990c87240262bd5b6da4b8166a6205b16c870334ce8fd5`;
its runtime readback authenticated solver revision `d4e78cf`, 8 requested and
effective cores, Slurm task 96338, job 840582, and a clean solver tree before
building `maxwell_matrix`. The older digest occurs zero times in the corrected
payload. The corrected lane completed Maxwell matrix, capacitance, and loss
postprocessing and injected eighth-model losses of 114.77 W (Tx), 69.04 W
(Rx), and 388.24 W (core) into Icepak. Its fixed-boundary readback matched fan
1.5 m/s, TIM conductivity 0.2 W/(m K), and both 2 mm pads. It authenticated 15
native mesh operations over 35 assigned objects with no required thin object
missing, confirmed that the standalone `GenerateMesh` call was omitted, passed
the explicit direct-Analyze gate, and dispatched native
`Analyze("ThermalSetup")`. No temperature PASS is claimed until that solve
terminates and its result is collected. Task 96338 is therefore the effective
candidate-5 fast lane.

Task 96337's exact Scheduler terminal GET was sealed as a pre-EM operational
failure and excluded from selection and NDS. A dedicated task-96338 collector
now polls Scheduler GET at 30 s intervals and is authorized to collect only the
exact n115/allocation-14650/job-840582/same-node-96332 lineage. It performs no
Scheduler POST or cancellation. The seven-lane post-success watcher and local
selector consume this direct-lane collection schema, while automatic Full
launch remains disabled.

## Web UI and final drawing follow-on

The live UI at `http://127.0.0.1:8010/` now shows task 96338 as the effective
candidate-5 lane, tasks 96332 and 96337 as lifecycle-visible but superseded,
the task-96337 pre-EM operational failure, actual scientific/production PASS
counts of zero, and automatic Full disabled. The API readback is available,
integrity-verified, and non-stale. A v6 cutover evidence-count overflow was
detected and corrected without stopping the UI listener.

After a final design is selected, the drawing package must use the PDF and
PPTX named `설계도면260706` under
`Z:\Projects\2025\8. 한양대 - 선박용MVDC\설계도면` as the layout and
annotation reference. The model views must come from the final geometry with
rounded winding corners. Reference inspection runs in parallel with FEA; no
source drawing file is to be modified.

## Symmetry-to-Full evidence boundary

No contract-valid, identical-geometry Standard/Complete-Full thermal pair was
found for the current 1.5 m/s cooling condition. The user's expectation that
thermal differences are small is therefore an execution policy, not an
authenticated temperature correction.

Two older identical-geometry electromagnetic pairs at fan 6 m/s showed
half-magnetizing resonance changes of +0.1505% and +0.1931%, and modeled total
loss changes of -4.38% and -4.82% from Standard to Full. Leakage-related
quantities differed by about 9--10%. Their Full thermal extraction was invalid,
and their TIM/fan contract differs from this campaign, so none of their
temperatures may be reused. The current policy remains: select with actual
symmetric FEA and run Full only once for the final selected design.

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
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_postsuccess_96325_96327_96328_96330_96331_96332_96333_96337_96338_v6\state.json`
- Official 8 n111 failover final seal:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_official8_failover_622097dde126_n111_260726_v1\submission\final_seal.json`
- Candidate-5 direct-Analyze v1 operational-failure submission seal:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_official5_direct_analyze_samenode_v1\submission\final_seal.json`
- Candidate-5 corrected same-node direct-Analyze submission seal:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_official5_direct_analyze_samenode_r1_v2\submission\final_seal.json`
- Candidate-5 task-96337 failure ledger and task-96338 GET-only collector root:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\postdeadline_standard_official5_direct_analyze_retry_task96338_v1`
- Symmetric-first local trust waiting manifest:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\local_trust_waiting_20260726_v2\acquisition_manifest.json`
- Live UI:
  `http://127.0.0.1:8010/`

The Scheduler repository remains a separate project. No Scheduler source was
mixed into the MFT repository for these submissions.
