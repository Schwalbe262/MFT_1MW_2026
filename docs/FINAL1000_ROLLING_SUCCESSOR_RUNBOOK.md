# final1000 cancellation-free successor rollout

This runbook first rolls the active `mft-t1fg-*` search to the 4-CPU resource
and 200/160/90/50 quota policy while reusing the exact current stage bundles.
That fast resource cutover does not wait for a new smoke receipt.  A second,
identical cancellation-free handoff can later dual-bind the negative-loss
patched bundles after their replay and publication evidence is complete.

## Sealed successor policy

- scheduler profile: `standard`
- CPU: 4 per task
- memory: 28,672 MiB per task
- `max_workers_per_node`: 32
- priority: 1
- surrogate inference threads: 8 (science/runtime identity is unchanged)
- logical active target: 500 = 200 + 160 + 90 + 50
- refill: smooth weighted-deficit round-robin, never stage-block append
- scheduler mutation allow-list: `POST /api/tasks` only
- cancellation, preemption, AEDT, and FEA: prohibited

For `standard` tasks the deployed scheduler applies
`max_workers_per_node` as an allocation-level upper bound.  Its ordinary CPU
and memory free-fit calculation remains the hard capacity gate.  The value 32
therefore avoids the predecessor's eight-worker allocation throttle without
bypassing 4-CPU or 28-GiB accounting.  This is also the previously proven
4c/28GiB/inference-8 production identity (2,606 completed tasks).

## Phase 1: immediate resource/quota successor

1. Render the successor launch plan from the **current** sealed bindings with
   the release containing `tier1_final1000_slurm_launch.py` from this change.
2. Confirm the launch plan reports exactly `4 / 28672 / 32 / priority 1` and
   every payload reports `inference_threads=8`.
3. Confirm its operational quotas are exactly 200/160/90/50.  The stage
   science profiles and all four bundle/READY identities must remain exact.
4. Keep the predecessor running until this local plan is fully validated.

Render from the currently deployed bindings (local write only):

```powershell
python tools\tier1_final1000_slurm_launch.py render `
  --bindings C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\stage-bindings-module.json `
  --output C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\launch-plan-resource4-quota-v1.json
```

Validation is read-only:

```powershell
python tools\tier1_final1000_slurm_launch.py validate `
  --plan C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\launch-plan-resource4-quota-v1.json
```

## Freeze the predecessor controller, not its tasks

The current controller already watches this exact stop file:

```powershell
New-Item -ItemType File -Force `
  C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\STOP_FINAL1000_CONTROLLER_MODULE
```

Wait for the predecessor controller process to exit normally.  Verify its last
result has `stop_requested=true`, `running_tasks_cancelled=false`, and verify
the sealed state has `stop_requested=true`.  Do not kill the controller and do
not cancel any queued/running task.  Once frozen, no process may write the
predecessor state.

## Fail-closed migration preflight

First run without `--apply`.  This performs scheduler GETs and remote READY
reads but writes neither scheduler state nor the successor state file.

```powershell
python tools\tier1_final1000_rolling_migration.py `
  --predecessor-plan C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\launch-plan-module.json `
  --predecessor-state C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\controller\state-module.json `
  --successor-plan C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\launch-plan-resource4-quota-v1.json `
  --successor-state C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\controller\state-resource4-quota-v1.json `
  --scheduler-url http://127.0.0.1:8002 `
  --resource-quota-only
```

Required output includes `scheduler_post_count=0`, the complete predecessor
entry count, the observed active count by stage, the unchanged next-seed
cursors, and `cancellation_performed=false`.  Any state drift, missing task,
task ID/name/dedupe/resource mismatch, duplicate seed, READY mismatch, reused
bundle identity, or foreign mutation endpoint aborts the preflight.

Run the same command once with `--apply`.  The only write is an atomic local
successor state file.  A crash before `os.replace` leaves no partial state;
rerun the command.  An existing state file is accepted only if byte-equivalent
in identity to the newly derived state.

## Start the successor controller

Start exactly one controller from the successor immutable release using the
successor launch plan and migrated state:

```powershell
python tools\tier1_final1000_slurm_controller.py `
  --plan C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\launch\launch-plan-resource4-quota-v1.json `
  --state C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\controller\state-resource4-quota-v1.json `
  --scheduler-url http://127.0.0.1:8002 `
  --apply --watch --poll-seconds 15 `
  --stop-file C:\path\to\STOP_FINAL1000_SUCCESSOR
```

The imported active tasks keep running untouched.  The new controller submits
only a gap created when one naturally becomes terminal.  The old distribution
64/96/128/212 is allowed temporarily: close/final excess drains naturally and
weighted-deficit refill transfers those slots into entry/bridge until the live
distribution reaches 200/160/90/50.  Total logical active remains 500 after
every successful reconciliation cycle; no per-stage excess is cancelled.

Monitor these fields in each controller result:

- `active_count=500`
- `active_count_by_stage` converging naturally to 200/160/90/50
- `rolling_migration=true`
- `successor_canary_task_ids_by_stage`
- `successor_canary_status_by_stage`
- `cancellation_performed=false`
- submitted tasks use 4 CPU, 28,672 MiB, max-workers 32, priority 1

If any fail-closed check fires, leave all Slurm tasks untouched, stop only the
successor controller through its stop file, preserve both state files and
logs, and diagnose the identity mismatch.  Never use cancellation as rollback.

## Phase 2: dual-bind patched bundles later

After negative-loss quarantine exact replay and all four immutable patched
bundle publications pass, render another 4c/28GiB/max32 plan from those new
bindings.  Gracefully stop the phase-1 controller through its own stop file.
This quota-optimized release accepts the stopped phase-1 predecessor only at
its historically sealed 160/140/120/80 quotas and accepts the patched
successor only at 200/160/90/50.  Neither side accepts an arbitrary quota
mapping.  The differing policies are authenticated independently before any
scheduler inventory GET.
Then invoke `tier1_final1000_rolling_migration.py` again with:

- predecessor plan/state = the stopped phase-1 resource/quota plan and state;
- successor plan/state = the new patched plan and a new state path;
- no `--resource-quota-only` flag (default mode is `patched_bundle`).

The tool authenticates both legacy-8c and phase-1-4c entries in the mixed
ledger, imports all of them, preserves the next-seed cursors, and resets only
the patched-bundle canary gate.  Starting the patched controller again replaces
natural terminal gaps only.  Old and patched bundle tasks may coexist safely;
no running or queued task is cancelled or rewritten.

## Chained patched/science successor (v2 cohort catalog)

An already patched `200/160/90/50` controller can roll again without dropping
its older terminal ledger. Supply the current predecessor plan normally and
repeat `--ancestor-plan` once for every older immutable launch plan named by
the predecessor state's sealed `harvest_cohorts` catalog. The migration
reproduces every old task using its own 8c/4c resource policy and 200/300
generation science identity. Missing, extra, or mismatched ancestor plans
fail before any successor state is written.

While the predecessor is still running, create read-only planning evidence:

```powershell
python tools\tier1_final1000_rolling_migration.py `
  --predecessor-plan <current-gen200-plan.json> `
  --ancestor-plan <older-cohort-plan.json> `
  --predecessor-state <live-gen200-state.json> `
  --successor-plan <gen300-plan.json> `
  --successor-state <future-gen300-state.json> `
  --scheduler-url http://127.0.0.1:8002 `
  --allow-running-predecessor-shadow `
  --shadow-state-evidence-output <sealed-shadow-state-evidence.json> `
  --evidence-output <sealed-local-dry-run-receipt.json>
```

This mode rejects `--apply`, never writes the successor state, marks the
in-memory state `shadow_only=true`, and the successor controller rejects that
shadow as not cutover-ready. The optional shadow-state evidence is an atomic,
SHA-sealed local artifact intended only for the harvester dry-run below; the
normal controller validator also rejects it. It reports namespace tasks
outside the ledger; they are never imported.

Authenticate and project all three current cohorts without writing indexes:

```powershell
python tools\tier1_final1000_slurm_harvest.py `
  --launch-plan <gen300-plan.json> `
  --bindings <gen300-stage-bindings.json> `
  --predecessor-launch-plan <current-gen200-plan.json> `
  --predecessor-bindings <current-gen200-stage-bindings.json> `
  --ancestor-launch-plan <older-gen200-plan.json> `
  --ancestor-bindings <older-gen200-stage-bindings.json> `
  --controller-state <sealed-shadow-state-evidence.json> `
  --allow-chained-shadow-state `
  --runtime <final1000-runtime> `
  --scheduler-url http://127.0.0.1:8002 `
  --protected-current7-index <current7-primary-index.json>
```

This is a single dry-run only. `--allow-chained-shadow-state` is incompatible
with both `--apply` and `--watch`. It performs scheduler GET and optional SFTP
reads, but scheduler mutation, remote write, local index write, AEDT, and FEA
counts remain zero. All catalog members, plans, manifests, READY receipts,
ledger entries, fixed-generation science identities, and task resource seals
must authenticate. The older plan/binding pair is repeated once per ancestor;
missing or excessive pairs fail closed. The legacy v1 two-role form remains
supported and explicitly rejects ancestor arguments.

For the real handoff, gracefully stop the predecessor first, omit
`--allow-running-predecessor-shadow`, run the command once without `--apply`,
then once with `--apply`. The v2 successor state imports every historical and
active entry under an append-only cohort id. All imported entries become
`origin=predecessor`; the four new successor canaries start only as natural
terminal gaps appear. The controller still has no cancellation or preemption
method.

Exactly one successor canary is permitted per stage. If more than four natural
gaps exist before all four remote preflights pass, only the four stage canaries
are submitted and every extra gap is held empty. Thus active count may be
temporarily below 500 during this bounded safety gate. No extra task may be
labelled canary and no refill may bypass the gate. After all four pass, weighted
deficit refill restores exact active 500. A failed or terminal canary is not
silently replaced; operator diagnosis is required. This policy is sealed as
`successor_canary_gap_policy` in the v2 migration state and controller action
evidence.

## Keep one mixed-ledger UI projection during both phases

The v1 rolling state seals a fixed two-cohort harvest contract. The v2 state
uses an append-only multi-cohort catalog, without copying plans, manifests, or
per-task envelopes into growing migration metadata. Run the harvester with the
successor pair, current predecessor pair, and one repeated ancestor pair for
every older catalog plan for the entire natural drain:

```powershell
python tools\tier1_final1000_slurm_harvest.py `
  --launch-plan <successor-plan.json> `
  --bindings <successor-stage-bindings.json> `
  --predecessor-launch-plan <stopped-predecessor-plan.json> `
  --predecessor-bindings <stopped-predecessor-stage-bindings.json> `
  --ancestor-launch-plan <older-plan.json> `
  --ancestor-bindings <older-stage-bindings.json> `
  --controller-state <successor-state.json> `
  --runtime C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721 `
  --scheduler-url http://127.0.0.1:8002 --apply --watch --poll-seconds 30
```

In phase 1 the two roles share bundle IDs but authenticate different 8-core
and 4-core resource envelopes.  In phase 2 the predecessor role may contain
both resource policies under the old bundle IDs while the successor role uses
the patched bundle IDs.  Every terminal result is checked against the
manifest/READY receipt belonging to its ledger origin.  The harvester then
publishes one combined index per condition with explicit
`source_bundle_cohorts`, `mixed_bundle_projection`, and
`mixed_resource_policy_projection` evidence.

A scheduler HTTP 429 while reading a canary status is retried three times with
a bounded 0.25/0.5-second backoff, then treated only as
`remote_preflight_status_read_busy` until the next controller poll. It never
passes the gate or terminates the watch loop; all non-busy identity and content
errors remain fail-closed.

The already-running phase-1 state created before `harvest_cohorts` was added
is supported without rewriting it.  Compatibility derivation is allowed only
for `resource_quota_only + legacy_8c`, where predecessor and successor share
the exact four bundle bindings.  The harvester still requires both plan and
binding pairs and checks the predecessor plan SHA before accepting that
derivation.  A catalog-less patched-bundle state is always rejected.
