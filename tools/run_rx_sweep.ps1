$ErrorActionPreference = "Stop"

$cases = @(
    @{ seq = "1"; name = "case_1_gap10"; side = "1.0"; gap = "10"; width = "1.0" },
    @{ seq = "2"; name = "case_2_gap20"; side = "1.0"; gap = "20"; width = "1.0" },
    @{ seq = "3"; name = "case_3_gap30"; side = "1.0"; gap = "30"; width = "1.0" },
    @{ seq = "4"; name = "case_4_gap40"; side = "1.0"; gap = "40"; width = "1.0" }
)

foreach ($c in $cases) {
    Write-Host "=== $($c.name) (seq=$($c.seq), side=$($c.side), gap=$($c.gap), width=$($c.width)) ==="
    $env:SNAPSHOT_SEQ = $c.seq
    $env:SNAPSHOT_CASE = $c.name
    $env:RX_SIDE_OFFSET_FACTOR = $c.side
    $env:SIDE_CORE_SURFACE_GAP = $c.gap
    $env:RX_SIDE_SPACE_WIDTH_FACTOR = $c.width

    & "C:\Users\peets\anaconda3\Scripts\conda.exe" run -n pyaedt2026v1 python "Y:\git\MFT_1MW_2026\tools\run_modeling_snapshot.py"
}
