param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8010,
    [ValidateSet("127.0.0.1", "0.0.0.0")]
    [string]$ListenAddress = "127.0.0.1",
    [string[]]$OperatorHosts = @(),
    [switch]$NoBrowser,
    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$MonitorDir = Join-Path $RepoRoot "regression_260707\monitoring"
$VenvDir = Join-Path $MonitorDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $MonitorDir "requirements.txt"
$HashMarker = Join-Path $VenvDir ".requirements.sha256"
$RuntimeDir = Join-Path $MonitorDir "runtime"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "[MFT monitor] Creating isolated Python environment..." -ForegroundColor Cyan
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $PyLauncher) {
        throw "Python launcher 'py' was not found. Install Python 3.11+ first."
    }
    & $PyLauncher.Source -3.11 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the monitoring virtual environment." }
    $ForceInstall = $true
}

$RequirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Requirements).Hash
$InstalledHash = if (Test-Path -LiteralPath $HashMarker) { (Get-Content -LiteralPath $HashMarker -Raw).Trim() } else { "" }
if ($ForceInstall -or $InstalledHash -ne $RequirementsHash) {
    Write-Host "[MFT monitor] Installing WEB UI dependencies..." -ForegroundColor Cyan
    & $Python -m pip install --disable-pip-version-check -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install monitoring dependencies." }
    Set-Content -LiteralPath $HashMarker -Value $RequirementsHash -Encoding ascii
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$StdoutLog = Join-Path $RuntimeDir "server.stdout.log"
$StderrLog = Join-Path $RuntimeDir "server.stderr.log"
$Url = "http://127.0.0.1:$Port"
if ($ListenAddress -eq "0.0.0.0" -and $OperatorHosts.Count -eq 0) {
    throw "-OperatorHosts is required when listening on the trusted LAN."
}
$Arguments = @(
    "-m", "uvicorn", "regression_260707.monitoring.app:app",
    "--host", $ListenAddress, "--port", $Port.ToString()
)

Write-Host "[MFT monitor] Starting $Url" -ForegroundColor Green
$PreviousOperatorHosts = $env:MFT_MONITOR_OPERATOR_HOSTS
try {
    if ($OperatorHosts.Count -gt 0) {
        $env:MFT_MONITOR_OPERATOR_HOSTS = $OperatorHosts -join ","
    }
    $Server = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $RepoRoot `
        -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog -WindowStyle Hidden -PassThru
} finally {
    $env:MFT_MONITOR_OPERATOR_HOSTS = $PreviousOperatorHosts
}

try {
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
        if ($Server.HasExited) {
            throw "Monitoring server exited early. See $StderrLog"
        }
        try {
            $Response = Invoke-WebRequest -UseBasicParsing -Uri "$Url/healthz" -TimeoutSec 1
            if ($Response.StatusCode -eq 200) { $Ready = $true; break }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $Ready) { throw "Monitoring server did not become ready. See $StderrLog" }
    Write-Host "[MFT monitor] Ready. Press Ctrl+C to stop. Logs: $RuntimeDir" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $Url }
    Wait-Process -Id $Server.Id
} finally {
    if (-not $Server.HasExited) { Stop-Process -Id $Server.Id -ErrorAction SilentlyContinue }
}
