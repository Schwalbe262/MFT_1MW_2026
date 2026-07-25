# MFT SAFE REFILL and terminal-watcher restart

This procedure changes only the local watcher process. It never cancels or
otherwise mutates a Scheduler task.

## Preconditions

1. Use the committed SAFE REFILL revision containing
   `mft_goal_safe_refill.py` and the dynamic watcher extension loader.
2. Authenticate `safe_refill_plan.json` with a dry-run `cycle` first.
3. Authenticate `watcher_extension_authority.json`. It must bind the existing
   immutable seven-slot `watch_plan.json`, the exact SAFE REFILL plan, and only
   logical slots 96223, 96224, and 96230.
4. Confirm the existing watcher PID from `watcher.pid.json` and its command
   line. Do not infer or use a stale PID.

## Safe local restart

Run these PowerShell operations from the committed worktree. Replace the three
paths with the authenticated runtime paths.

```powershell
$runtimeRoot = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\terminal_success_watcher_c276213_260725'
$watchPlan = Join-Path $runtimeRoot 'watch_plan.json'
$extensionAuthority = 'C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\safe_refill_4fac_c276_260725\watcher_extension_authority.json'
$pythonExe = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'

$pidReceipt = Get-Content -Raw (Join-Path $runtimeRoot 'watcher.pid.json') |
    ConvertFrom-Json
$oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($pidReceipt.pid)"
if (-not $oldProcess -or
    $oldProcess.CommandLine -notlike '*mft_goal_terminal_success_watcher.py*') {
    throw 'watcher PID/command identity is not authenticated'
}

Stop-Process -Id $pidReceipt.pid
Wait-Process -Id $pidReceipt.pid -ErrorAction SilentlyContinue

$stdout = Join-Path $runtimeRoot 'logs\watcher.stdout.log'
$stderr = Join-Path $runtimeRoot 'logs\watcher.stderr.log'
$arguments = @(
    'tools\mft_goal_terminal_success_watcher.py',
    'run',
    '--watch-plan', $watchPlan,
    '--extension-authority', $extensionAuthority
)
$process = Start-Process -FilePath $pythonExe -ArgumentList $arguments `
    -WorkingDirectory '<committed-safe-refill-worktree>' `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -PassThru -WindowStyle Hidden
```

The stop above terminates only the local GET-only watcher. It does not call a
Scheduler endpoint. The operating-system lock is released automatically when
the old process exits.

## Post-restart proof

1. `watcher.pid.json` contains the new PID and the exact extension-authority
   file record.
2. The process command line contains both `--watch-plan` and
   `--extension-authority`.
3. The first cycle completes with an empty stderr log.
4. Before any refill submission, the watcher still reports the seven base
   slots and zero authorized extensions.
5. After a successful guarded refill, the producer first writes an immutable
   timeout12h submission receipt and then an immutable extension receipt.
   Within one polling interval the watcher reauthenticates the authority,
   refill plan, source plan, submission, fixed boundary identity, and exact
   logical/execution IDs before adding the slot.

If extension authentication fails, the watcher fails closed before processing
that cycle. Do not start SAFE REFILL with `--authorize-submit` until the
restarted watcher has passed these checks and root has explicitly approved the
live enable.
