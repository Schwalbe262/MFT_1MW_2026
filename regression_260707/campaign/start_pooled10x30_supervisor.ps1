param(
    [Parameter(Mandatory = $true)]
    [switch]$Execute,

    [string]$Python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe',
    [string]$CodeRoot = 'C:\Users\peets\codex_worktrees\mft_q23_same_node_clean_260716',
    [string]$RuntimeDir = 'C:\Users\peets\slurm_scheduler_runtime\mft_pooled10x30',
    [string]$DatasetDir = 'Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset',
    [string]$SchedulerUrl = 'http://127.0.0.1:8002',
    [string]$PoolUrl = 'http://172.16.10.37:18790',
    [int]$IntervalSeconds = 15,
    [int]$RestartDelaySeconds = 15
)

$ErrorActionPreference = 'Stop'
if (-not $Execute) {
    throw 'Explicit -Execute acknowledgement is required.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
$Controller = Join-Path $CodeRoot 'regression_260707\campaign\pooled10x30_soak.py'
if (-not (Test-Path -LiteralPath $Controller -PathType Leaf)) {
    throw "Controller not found: $Controller"
}
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$LogPath = Join-Path $RuntimeDir 'supervisor.log'
$PidPath = Join-Path $RuntimeDir 'supervisor.pid'
$PID | Set-Content -LiteralPath $PidPath -Encoding ascii
$Arguments = @(
    '-u', $Controller,
    '--execute',
    '--authorize-pooled10-aedt-30-projects',
    '--interval-seconds', "$IntervalSeconds",
    '--runtime-dir', $RuntimeDir,
    '--dataset-dir', $DatasetDir,
    '--scheduler-url', $SchedulerUrl,
    '--pool-url', $PoolUrl
)

while ($true) {
    "[$(Get-Date -Format o)] starting bounded pooled10x30 controller" |
        Tee-Object -FilePath $LogPath -Append
    $ExitCode = 1
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $ExitCode = $LASTEXITCODE
    }
    catch {
        ($_ | Out-String) | Tee-Object -FilePath $LogPath -Append
        $ExitCode = 1
    }
    finally {
        $ErrorActionPreference = 'Stop'
    }
    "[$(Get-Date -Format o)] controller exited rc=$ExitCode; restarting in ${RestartDelaySeconds}s" |
        Tee-Object -FilePath $LogPath -Append
    Start-Sleep -Seconds $RestartDelaySeconds
}
