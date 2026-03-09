param(
    [string]$OutputPath = "Y:\git\MFT_1MW_2026\picture\latest.png"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$image = [System.Windows.Forms.Clipboard]::GetImage()
if ($null -eq $image) {
    Write-Error "Clipboard does not contain an image. Capture first (Win+Shift+S)."
    exit 1
}

$dir = Split-Path -Parent $OutputPath
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

$image.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$image.Dispose()

Write-Host "Saved clipboard image to: $OutputPath"
