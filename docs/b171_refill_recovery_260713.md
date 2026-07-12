# B171 refill recovery incident (2026-07-13)

## Observed failure

The reviewed no-autopause controller loaded Y state cycle 388 and acquired the
Y-only campaign mutation lock. Its first mature-production reconciliation took
about five minutes because the state cache had 1,288 terminal outcomes while
the scheduler returned 10,000 historical `mft-camp` terminals. Of the 8,712
uncached records, 528 were failed and 7,793 were cancelled.

Cycle 389 then reached the feeder with logical project active count 3, but the
capacity request omitted `project=MFT_1MW_2026v1`. License admission therefore
treated it as an unknown FEA project and returned `queue_state=blocked`. Printing
the mojibake em-dash in that queue reason through cp949 caused a
`UnicodeEncodeError`.

The failed-closed cycle is terminal and contains no scheduler mutation:

- `status=failed_closed`
- `events=[]`
- `planned_count=0`
- `submission_deficit=0`
- `submitted_count=0`
- no new `mft-camp` scheduler task

The three priority-10 1K101 tasks remained running and no existing task was
cancelled.

## Bounded corrections

1. `feeder.scheduler_snapshot` now includes the exact MFT project in
   `/api/task-capacity`, so license admission evaluates the configured project
   before priority and attachment.
2. `inspect_production_tasks` continues to validate every completed task through
   its actual solver result. A scheduler `failed` or `cancelled` status remains
   terminal-invalid and is now classified from scheduler metadata without an
   unbounded remote stderr scan. Explicit incident-diagnostic callers can still
   request bounded stderr enrichment through `_refresh_failure_outcome`.
3. The Y runtime launcher sets UTF-8 and authenticates the reviewed source
   hashes before mutation. State, dataset, loop lock, and campaign mutation lock
   remain on Y.

No status was manually converted to valid and no outcome cache was manually
backfilled.

## Verification

- controller release/main pre-patch restored files: SHA/length 9/9 identical
- required controller suites: 31/31 pass
- rapid-campaign reconciliation suite: 31/31 pass
- project-aware capacity regression: 1/1 pass
- static audit: `scheduler_query_count=0`, `scheduler_mutation_count=0`
- project contract: `MFT_1MW_2026v1`, `max_active_tasks=300`
- forbidden pool400 CLI test remains passing

## Activation and rollback

Before restart, the controller must reconcile the terminal failed-closed cycle
389. It then re-reads the live project deficit inside the Y mutation lock and
may submit only the difference to 300. Existing running tasks are never a
cancellation target.

Rollback is process-scoped: verify PID start time and command contain
`run_controller_6a870.py`, then stop only that PID. Do not cancel Slurm tasks,
do not stop the collector/checkpoint, and do not reset the dirty main worktree.
Source rollback uses a reviewed follow-up commit from the clean release branch;
it is never implemented by broad checkout/reset on main.
