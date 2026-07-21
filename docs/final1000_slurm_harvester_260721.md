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
- A completed task with invalid remote evidence is recorded as a refusal and
  cannot contribute a candidate.
- Failed/cancelled/timeout tasks remain visible in the scheduler inventory but
  never become authenticated design records.
- Completed task data and terminal scheduler envelopes are cached locally and
  SHA-replayed, avoiding repeated scheduler and SFTP reads on later polls.
- No AEDT or FEA method is imported or invoked.
