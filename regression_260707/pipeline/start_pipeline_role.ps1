[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Controller", "Supervisor")]
    [string]$Role,

    [Parameter(Mandatory = $true)]
    [string]$SolverRevision,

    [Parameter(Mandatory = $true)]
    [string]$LibraryRevision,

    [Parameter(Mandatory = $true)]
    [string]$VerificationConfig,

    [Parameter(Mandatory = $true)]
    [string]$ReviewedVerificationConfigSha256,

    [string]$PipelineRuntimeRoot = "C:\Users\peets\slurm_scheduler_runtime\mft_pipeline",
    [string]$MftRuntimeRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = "C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe",
    [string]$Dataset,
    [string]$Registry,
    [ValidateRange(30, 86400)]
    [int]$ControllerIntervalSeconds = 600,
    [ValidateRange(1, 100000)]
    [int]$OptunaTrials = 200,
    [switch]$RestartOnFailure,
    [ValidateRange(1, 3600)]
    [int]$RestartDelaySeconds = 30,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-ExistingPath([string]$Value, [string]$Label, [string]$PathType) {
    if (-not (Test-Path -LiteralPath $Value -PathType $PathType)) {
        throw "$Label is unavailable: $Value"
    }
    return (Resolve-Path -LiteralPath $Value).ProviderPath
}

function Assert-FullRevision([string]$Value, [string]$Label) {
    if ($Value -notmatch "^[0-9a-fA-F]{40}$") {
        throw "$Label must be a full 40-character SHA"
    }
    return $Value.ToLowerInvariant()
}

function Write-LauncherEvent([string]$Path, [string]$Name, [hashtable]$Fields) {
    $record = [ordered]@{
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
        event = $Name
        role = $Role.ToLowerInvariant()
        launcher_pid = $PID
    }
    foreach ($entry in $Fields.GetEnumerator()) {
        $record[$entry.Key] = $entry.Value
    }
    ($record | ConvertTo-Json -Compress -Depth 8) | Add-Content -LiteralPath $Path -Encoding UTF8
}

function Install-SealedFile([string]$Source, [string]$Destination, [string]$ExpectedSha256) {
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        $temporary = "$Destination.$([Guid]::NewGuid().ToString('N')).tmp"
        [IO.File]::Copy($Source, $temporary, $false)
        try {
            [IO.File]::Move($temporary, $Destination)
        }
        catch {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
                throw
            }
        }
    }
    $installedSha = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($installedSha -ne $ExpectedSha256) {
        throw "sealed verification config hash mismatch: $Destination"
    }
}

$solver = Assert-FullRevision $SolverRevision "solver revision"
$library = Assert-FullRevision $LibraryRevision "library revision"
if ($ReviewedVerificationConfigSha256 -notmatch "^[0-9a-fA-F]{64}$") {
    throw "reviewed verification config SHA256 must have 64 hexadecimal characters"
}
$reviewedConfigSha = $ReviewedVerificationConfigSha256.ToLowerInvariant()

$pythonPath = Resolve-ExistingPath $Python "Python" "Leaf"
$mftRoot = Resolve-ExistingPath $MftRuntimeRoot "MFT runtime root" "Container"
if (-not (Test-Path -LiteralPath (Join-Path $mftRoot "pipeline\__main__.py") -PathType Leaf)) {
    throw "MFT runtime root does not contain the durable pipeline package: $mftRoot"
}
$configPath = Resolve-ExistingPath $VerificationConfig "verification config" "Leaf"
$actualConfigSha = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualConfigSha -ne $reviewedConfigSha) {
    throw "verification config does not match the reviewed SHA256"
}

$verification = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
$stageNames = @($verification.PSObject.Properties.Name)
if ($stageNames.Count -ne 2 -or $stageNames -notcontains "standard" -or $stageNames -notcontains "fine") {
    throw "verification config must contain exactly the standard and fine stages"
}
foreach ($stageName in @("standard", "fine")) {
    $stage = $verification.$stageName
    if ($stage.adapter -ne "mft_scheduler_v1" -or $stage.execute -ne $true) {
        throw "$stageName verification must explicitly enable mft_scheduler_v1"
    }
    if (-not $stage.library_root) {
        throw "$stageName verification library_root is required"
    }
    if (-not [IO.Path]::IsPathRooted([string]$stage.library_root)) {
        throw "$stageName verification library_root must be absolute"
    }
    [void](Resolve-ExistingPath ([string]$stage.library_root) "$stageName verification library_root" "Container")
}

$stateRoot = [IO.Path]::GetFullPath($PipelineRuntimeRoot)
$trimCharacters = [char[]]@(
    [IO.Path]::DirectorySeparatorChar,
    [IO.Path]::AltDirectorySeparatorChar
)
$mftPrefix = $mftRoot.TrimEnd($trimCharacters) + [IO.Path]::DirectorySeparatorChar
if ($stateRoot.Equals($mftRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $stateRoot.StartsWith($mftPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "pipeline runtime root must be outside the MFT source/runtime tree"
}

$logsRoot = Join-Path $stateRoot "logs"
$configRoot = Join-Path $stateRoot "config"
[IO.Directory]::CreateDirectory($stateRoot) | Out-Null
[IO.Directory]::CreateDirectory($logsRoot) | Out-Null
[IO.Directory]::CreateDirectory($configRoot) | Out-Null
$sealedConfig = Join-Path $configRoot "verification-$reviewedConfigSha.json"
Install-SealedFile $configPath $sealedConfig $reviewedConfigSha

$roleName = $Role.ToLowerInvariant()
$stdoutLog = Join-Path $logsRoot "$roleName.stdout.log"
$stderrLog = Join-Path $logsRoot "$roleName.stderr.log"
$launcherLog = Join-Path $logsRoot "$roleName.launcher.jsonl"
$repositoryRoot = Split-Path -Parent $mftRoot
$arguments = @(
    "-m", "regression_260707.pipeline",
    "--runtime-root", $mftRoot,
    "--pipeline-root", $stateRoot
)
if ($Role -eq "Controller") {
    $arguments += @(
        "control",
        "--solver-revision", $solver,
        "--library-revision", $library,
        "--verification-commands", $sealedConfig,
        "--interval-seconds", [string]$ControllerIntervalSeconds,
        "--optuna-trials", [string]$OptunaTrials
    )
    if ($Dataset) {
        $arguments += @("--dataset", [IO.Path]::GetFullPath($Dataset))
    }
    if ($Registry) {
        $arguments += @("--registry", [IO.Path]::GetFullPath($Registry))
    }
}
else {
    $arguments += "supervise"
}

$contract = [ordered]@{
    schema_version = 1
    role = $roleName
    python = $pythonPath
    repository_root = $repositoryRoot
    mft_runtime_root = $mftRoot
    pipeline_runtime_root = $stateRoot
    solver_revision = $solver
    library_revision = $library
    verification_config = $sealedConfig
    verification_config_sha256 = $reviewedConfigSha
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    singleton_lock = (Join-Path $stateRoot "locks\$roleName.lock")
    arguments = $arguments
}
if ($ValidateOnly) {
    $contract | ConvertTo-Json -Depth 8
    exit 0
}

$attempt = 0
while ($true) {
    Write-LauncherEvent $launcherLog "process_start" @{
        attempt = $attempt
        solver_revision = $solver
        library_revision = $library
        verification_config_sha256 = $reviewedConfigSha
    }
    $exitCode = 1
    Push-Location $repositoryRoot
    try {
        & $pythonPath @arguments 1>> $stdoutLog 2>> $stderrLog
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-LauncherEvent $launcherLog "process_exit" @{
        attempt = $attempt
        exit_code = $exitCode
    }
    if ($exitCode -eq 73) {
        exit 73
    }
    if ($exitCode -eq 0 -or -not $RestartOnFailure) {
        exit $exitCode
    }
    $attempt += 1
    Write-LauncherEvent $launcherLog "restart_wait" @{
        attempt = $attempt
        delay_seconds = $RestartDelaySeconds
    }
    Start-Sleep -Seconds $RestartDelaySeconds
}
