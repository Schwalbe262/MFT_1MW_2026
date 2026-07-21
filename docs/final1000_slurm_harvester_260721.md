# Final1000 isolated controller/harvester wiring

The final-goal campaign has one controller namespace and four read-only result
indexes.  It does not replace or mutate Current7.

## Runtime layout

Use this root:

```text
C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721
```

The launch integration should place the immutable inputs and controller state
at:

```text
launch\stage-bindings.json
launch\launch-plan.json
controller\state.json
```

The harvester creates only these new local namespaces:

```text
conditions\entry-1200-t125\canonical\current7-index.json
conditions\bridge-1150-t115\canonical\current7-index.json
conditions\close-1075-t107p5\canonical\current7-index.json
conditions\final-1000-t100\canonical\current7-index.json
canonical\condition-indexes.json
```

Each condition subtree has its own immutable cohorts, authenticated seed
cache, and terminal scheduler-envelope cache.  A stage has one exact hard
constraint identity, so results from the four contracts are never mixed.

## Long-running processes

Run the already-separate controller and harvester as independent processes.
The controller is the only process that may submit, and it only submits
`mft-t1fg-*` / `mft-tier1-final1000:*` tasks.  The harvester has no scheduler
mutation or remote-write method.

```powershell
$Root = 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721'
$Code = '<detached final1000 release containing the controller and harvester>'

python "$Code\tools\tier1_final1000_slurm_controller.py" `
  --plan "$Root\launch\launch-plan.json" `
  --state "$Root\controller\state.json" `
  --scheduler-url 'http://127.0.0.1:8002' `
  --apply --watch --poll-seconds 15

python "$Code\tools\tier1_final1000_slurm_harvest.py" `
  --launch-plan "$Root\launch\launch-plan.json" `
  --bindings "$Root\launch\stage-bindings.json" `
  --controller-state "$Root\controller\state.json" `
  --runtime $Root `
  --scheduler-url 'http://127.0.0.1:8002' `
  --accounts 'Y:\runtime\slurm_scheduler\config\accounts.yaml' `
  --scheduler-source 'Y:\git\slurm_scheduler' `
  --protected-current7-index 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling\canonical\current7-index.json' `
  --apply --watch --poll-seconds 30 `
  --stop-file "$Root\STOP_FINAL1000_HARVEST"
```

During either rolling handoff, keep the two immutable launch/binding inputs
explicit.  The unprefixed pair is always the current successor (and the UI
projection anchor); the prefixed pair is the stopped predecessor whose tasks
are still draining naturally:

```powershell
python "$Code\tools\tier1_final1000_slurm_harvest.py" `
  --launch-plan "$Root\launch\launch-plan-resource4-quota-v1.json" `
  --bindings "$Root\launch\stage-bindings-module.json" `
  --predecessor-launch-plan "$Root\launch\launch-plan-module.json" `
  --predecessor-bindings "$Root\launch\stage-bindings-module.json" `
  --controller-state "$Root\controller\state-resource4-quota-v1.json" `
  --runtime $Root --scheduler-url 'http://127.0.0.1:8002' `
  --accounts 'Y:\runtime\slurm_scheduler\config\accounts.yaml' `
  --scheduler-source 'Y:\git\slurm_scheduler' `
  --protected-current7-index 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling\canonical\current7-index.json' `
  --apply --watch --poll-seconds 30 `
  --stop-file "$Root\STOP_FINAL1000_HARVEST"
```

For the patched-bundle handoff, point the predecessor pair at the stopped
phase-1 4-core plan/bindings and the unprefixed pair at the patched plan and
bindings.  The sealed controller state contains exactly two bounded cohort
identities.  Each ledger entry is replayed against its originating resource
policy and bundle manifest/READY receipt; results from both cohorts are then
aggregated into the same per-condition UI index.  Omitting either predecessor
argument in migration mode fails closed.

The live phase-1 resource-only state created before the explicit cohort field
does not need an in-place rewrite.  The reader derives the two roles only when
the sealed transition is exactly `resource_quota_only` from `legacy_8c`, both
plans prove the same bundle bindings, and the supplied predecessor plan SHA
matches the migration seal.  This compatibility path is unavailable to a
patched-bundle transition.

Do not give the controller and harvester the same stop file.  Stopping refill
should not prevent a final result harvest.

## UI read-only condition configuration

Keep `MFT_TIER1_CURRENT7_INDEX` pointed at the existing Current7 primary.  Add
the following separate Windows `os.pathsep` (`;`) list:

```powershell
$env:MFT_TIER1_CURRENT7_CONDITION_INDEXES = @(
  "$Root\conditions\entry-1200-t125\canonical\current7-index.json"
  "$Root\conditions\bridge-1150-t115\canonical\current7-index.json"
  "$Root\conditions\close-1075-t107p5\canonical\current7-index.json"
  "$Root\conditions\final-1000-t100\canonical\current7-index.json"
) -join ';'
```

The four documents use the existing
`mft-tier1-current7-slurm-rolling-index-v1` wire and are explicitly
display-only.  Their authenticated resonance condition is the band
`15 kHz <= min(f_tx, f_rx) < 20 kHz`, with both the minimum and maximum
constraint names present.  `harvest_observed_at` changes on every successful
poll while the immutable cohort `status_event_at` changes only with scheduler
events.

## Failure behavior

- A missing/renamed/foreign scheduler task aborts that poll and leaves the
  previous indexes intact.
- CPU, memory, max-workers, priority, capability, environment, and remote
  bundle drift in a scheduler row aborts the poll before result projection.
- Legacy 8-core and successor 4-core terminal results use their respective
  strict task-policy validators.  A patched result is authenticated only by
  its own bundle manifest; the successor manifest is never reused for an old
  task.
- A completed task with invalid remote evidence is recorded as a refusal and
  cannot contribute a candidate.
- Failed/cancelled/timeout tasks remain visible in the scheduler inventory but
  never become authenticated design records.
- Completed task data and terminal scheduler envelopes are cached locally and
  SHA-replayed, avoiding repeated scheduler and SFTP reads on later polls.
- No AEDT or FEA method is imported or invoked.
