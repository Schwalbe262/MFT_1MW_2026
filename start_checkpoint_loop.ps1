param(
    [string]$RuntimeRoot = "Y:\git\MFT_1MW_2026\regression_260707",
    [string]$Dataset = "",
    [string]$OutputRoot = "",
    [string]$RunRoot = "",
    [string]$Profile = "",
    [string]$Thresholds = "",
    [int]$IntervalSeconds = 600,
    [string]$Python = "$HOME\anaconda3\envs\pyaedt2026v1\python.exe",
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$SolverRevision,
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$LibraryRevision,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
if ($IntervalSeconds -lt 30) {
    throw "IntervalSeconds must be at least 30"
}
$CodeRoot = Join-Path $PSScriptRoot "regression_260707"
$ScriptPath = Join-Path $CodeRoot "training\checkpoint_orchestrator.py"
$ContractScriptPath = Join-Path $CodeRoot "training\checkpoint_contract.py"
$BaseArguments = @($ScriptPath, "--runtime-root", $RuntimeRoot)
$BaseArguments += @("--solver-revision", $SolverRevision, "--library-revision", $LibraryRevision)
if ($Dataset) { $BaseArguments += @("--dataset", $Dataset) }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RuntimeRoot "training"
}
if (-not $Thresholds) {
    $Thresholds = Join-Path $CodeRoot "training\model_quality_thresholds.json"
}
$Thresholds = (Resolve-Path -LiteralPath $Thresholds).ProviderPath
$BaseArguments += @("--output-root", $OutputRoot, "--thresholds", $Thresholds)
if ($Profile) {
    $Profile = (Resolve-Path -LiteralPath $Profile).ProviderPath
    $BaseArguments += @("--profile", $Profile)
}
if ($Execute) { $BaseArguments += "--execute" }
$ExplicitRunRoot = $RunRoot
$RevisionKey = "$($SolverRevision.ToLowerInvariant())-$($LibraryRevision.ToLowerInvariant())"

while ($true) {
    $ContractArguments = @($ContractScriptPath, "--thresholds", $Thresholds)
    if ($Profile) { $ContractArguments += @("--profile", $Profile) }
    $ContractKey = (& $Python @ContractArguments | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $ContractKey -notmatch '^[0-9a-f]{16}$') {
        Write-Warning "unable to fingerprint the checkpoint training contract"
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    if ($ExplicitRunRoot) {
        $CycleRunRoot = $ExplicitRunRoot
    } else {
        $RevisionContractKey = "$RevisionKey-c$ContractKey"
        $CycleRunRoot = Join-Path $OutputRoot (Join-Path "checkpoint_runs" $RevisionContractKey)
    }
    $Arguments = $BaseArguments + @(
        "--run-root", $CycleRunRoot,
        "--expected-contract-key", $ContractKey
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "checkpoint attempt failed; persistent state will retry on the next cycle"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
