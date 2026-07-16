param(
    [Parameter(Mandatory = $true)]
    [switch]$ExecuteMftFamilyProduction,

    [string]$Python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe',
    [string]$StateDir = 'Y:\git\MFT_1MW_2026\regression_260707\campaign',
    [string]$DatasetDir = 'Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset',
    [string]$LibraryRoot = 'Y:\git\pyaedt_library_release_e6b9_260715',
    [string]$SchedulerUrl = 'http://127.0.0.1:8001',
    [string]$PoolUrl = 'http://172.16.10.37:18790',
    [ValidateSet(1, 2)]
    [int]$ManifestVersion = 1,
    [string[]]$EligibleAccounts = @('dhj02', 'harry261', 'jji0930'),
    [int]$IntervalSeconds = 5,
    [int]$RestartDelaySeconds = 15
)

$ErrorActionPreference = 'Stop'
if (-not $ExecuteMftFamilyProduction) {
    throw 'Explicit -ExecuteMftFamilyProduction acknowledgement is required.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$Controller = Join-Path $PSScriptRoot 'q22_bounded_soak.py'
$LogDir = Join-Path $StateDir 'q22_logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogPath = Join-Path $LogDir 'q22_bounded_soak_supervisor.log'

$Arguments = @(
    '-u', $Controller,
    '--execute-mft-family-production',
    '--interval-seconds', "$IntervalSeconds",
    '--state-dir', $StateDir,
    '--dataset-dir', $DatasetDir,
    '--library-root', $LibraryRoot,
    '--scheduler-url', $SchedulerUrl,
    '--pool-url', $PoolUrl,
    '--manifest-version', "$ManifestVersion"
)
foreach ($Account in $EligibleAccounts) {
    if ([string]::IsNullOrWhiteSpace($Account)) {
        throw 'EligibleAccounts cannot contain an empty account name.'
    }
    $Arguments += @('--eligible-account', $Account)
}

while ($true) {
    $Started = Get-Date -Format o
    "[$Started] starting q22 open-ended controller" | Tee-Object -FilePath $LogPath -Append
    $ExitCode = 1
    try {
        # Windows PowerShell can promote a native program's stderr stream to a
        # terminating NativeCommandError when ErrorActionPreference is Stop.
        # A controller gate failure must be logged and restarted; it must not
        # kill this long-lived supervisor.
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
    $Stopped = Get-Date -Format o
    "[$Stopped] controller exited rc=$ExitCode; restarting in ${RestartDelaySeconds}s" |
        Tee-Object -FilePath $LogPath -Append
    Start-Sleep -Seconds $RestartDelaySeconds
}
