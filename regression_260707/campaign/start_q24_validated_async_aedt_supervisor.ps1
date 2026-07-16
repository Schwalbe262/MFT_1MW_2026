param(
    [Parameter(Mandatory = $true)]
    [switch]$ExecuteMftFamilyProduction,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$SchedulerPackageRevision,

    [string]$SchedulerPackageSuccessor = '',

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, [int]::MaxValue)]
    [int]$AdoptBaselineSerial,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, [int]::MaxValue)]
    [int]$AdoptBaselineDatasetRows,

    [string]$Python = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe',
    [string]$StateDir = 'Y:\git\MFT_1MW_2026\regression_260707\campaign',
    [string]$DatasetDir = 'Y:\git\MFT_solver_pooled_260714\regression_260707\data\dataset',
    [string]$LibraryRoot = 'Y:\git\pyaedt_library_release_e6b9_260715',
    [string]$SchedulerUrl = 'http://127.0.0.1:8001',
    [string]$PoolUrl = 'http://172.16.10.37:18790',
    [string[]]$EligibleAccounts = @(
        'dhj02', 'harry261', 'jji0930', 'dw16', 'r1jae262'
    ),
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
if (
    -not [string]::IsNullOrWhiteSpace($SchedulerPackageSuccessor) -and
    $SchedulerPackageSuccessor -notmatch '^[0-9a-f]{40}$'
) {
    throw 'SchedulerPackageSuccessor must be a full lowercase commit SHA.'
}

$Controller = Join-Path $PSScriptRoot 'q24_validated_async_aedt_campaign.py'
$LogDir = Join-Path $StateDir 'q24_logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogPath = Join-Path $LogDir 'q24_validated_async_aedt_supervisor.log'
$Arguments = @(
    '-u', $Controller,
    '--execute-mft-family-production',
    '--scheduler-package-revision', $SchedulerPackageRevision,
    '--adopt-baseline-serial', "$AdoptBaselineSerial",
    '--adopt-baseline-dataset-rows', "$AdoptBaselineDatasetRows",
    '--interval-seconds', "$IntervalSeconds",
    '--state-dir', $StateDir,
    '--dataset-dir', $DatasetDir,
    '--library-root', $LibraryRoot,
    '--scheduler-url', $SchedulerUrl,
    '--pool-url', $PoolUrl
)
if (-not [string]::IsNullOrWhiteSpace($SchedulerPackageSuccessor)) {
    $Arguments += @(
        '--scheduler-package-successor', $SchedulerPackageSuccessor
    )
}
foreach ($Account in $EligibleAccounts) {
    if ([string]::IsNullOrWhiteSpace($Account)) {
        throw 'EligibleAccounts cannot contain an empty account name.'
    }
    $Arguments += @('--eligible-account', $Account)
}

while ($true) {
    "[$(Get-Date -Format o)] starting q24 validated-async AEDT controller" |
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
