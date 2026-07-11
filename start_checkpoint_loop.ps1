param(
    [string]$RuntimeRoot = "Y:\git\MFT_1MW_2026\regression_260707",
    [string]$Dataset = "",
    [string]$OutputRoot = "",
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
$ScriptPath = Join-Path $PSScriptRoot "regression_260707\training\checkpoint_orchestrator.py"
$Arguments = @($ScriptPath, "--runtime-root", $RuntimeRoot)
$Arguments += @("--solver-revision", $SolverRevision, "--library-revision", $LibraryRevision)
if ($Dataset) { $Arguments += @("--dataset", $Dataset) }
if ($OutputRoot) { $Arguments += @("--output-root", $OutputRoot) }
if ($Execute) { $Arguments += "--execute" }

while ($true) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "checkpoint attempt failed; persistent state will retry on the next cycle"
    }
    Start-Sleep -Seconds $IntervalSeconds
}
