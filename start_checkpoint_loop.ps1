param(
    [string]$RuntimeRoot = "Y:\git\MFT_1MW_2026\regression_260707",
    [string]$Dataset = "",
    [string]$OutputRoot = "",
    [int]$IntervalSeconds = 600,
    [string]$Python = "$HOME\anaconda3\envs\pyaedt2026v1\python.exe",
    [string]$SolverRevision = "2ac926f678d58c4ec42aa8536fe7b509b42727c0",
    [string]$LibraryRevision = "e6b9b9d20a832ff5c3f7ca97218737a0b8650781",
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
