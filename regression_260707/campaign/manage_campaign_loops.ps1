param(
    [switch]$VerifyOnly,
    [int]$CollectorWaitSeconds = 900
)

$ErrorActionPreference = "Stop"

function Get-CampaignProcesses {
    $all = @(Get-CimInstance Win32_Process)
    $feeders = @($all | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -match "(?i)(^|[\\/ ])feeder\.py( |$)" -and
        $_.CommandLine -match "(?i)--loop"
    })
    $collectors = @($all | Where-Object {
        $_.Name -match "^(bash|sh)\.exe$" -and
        $_.CommandLine -match "(?i)auto_collect(?:[0-9]*|_loop)\.sh|auto_checkpoint2?\.sh|collect_relaunch\.log|campaign[\\/]collect_wave\.py"
    })
    $collectorLeaves = @($collectors | Where-Object {
        $_.CommandLine -match "(?i)auto_collect(?:[0-9]*|_loop)\.sh|auto_checkpoint2?\.sh" -and
        $_.CommandLine -notmatch "(?i)[ ]-c[ ]"
    })
    $activeCollectors = @($all | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -match "(?i)campaign[\\/]collect_wave\.py|collect_wave\.py[ ]+--prefix[ ]+mft-camp"
    })
    return @{
        All = $all
        Feeders = $feeders
        Collectors = $collectors
        CollectorLeaves = $collectorLeaves
        ActiveCollectors = $activeCollectors
    }
}

function Get-RootCount([object[]]$Processes) {
    $ids = @($Processes | Select-Object -ExpandProperty ProcessId)
    return @($Processes | Where-Object { $ids -notcontains $_.ParentProcessId }).Count
}

function Get-DescendantIds([int]$RootId, [object[]]$Processes) {
    $ids = @($RootId)
    $queue = @($RootId)
    while ($queue.Count -gt 0) {
        $parentId = $queue[0]
        $queue = @($queue | Select-Object -Skip 1)
        $children = @($Processes | Where-Object { $_.ParentProcessId -eq $parentId })
        foreach ($child in $children) {
            if ($ids -notcontains $child.ProcessId) {
                $ids += $child.ProcessId
                $queue += $child.ProcessId
            }
        }
    }
    return $ids
}

$snapshot = Get-CampaignProcesses
if ($VerifyOnly) {
    $feederRoots = Get-RootCount $snapshot.Feeders
    $collectorRoots = Get-RootCount $snapshot.CollectorLeaves
    $collectorTreeIds = @()
    foreach ($collector in $snapshot.Collectors) {
        $collectorTreeIds += Get-DescendantIds $collector.ProcessId $snapshot.All
    }
    $collectorTreeIds = @($collectorTreeIds | Sort-Object -Unique)
    $orphanCollectors = @($snapshot.ActiveCollectors | Where-Object {
        $collectorTreeIds -notcontains $_.ProcessId
    })
    Write-Output (
        "feeder_roots=$feederRoots collector_roots=$collectorRoots " +
        "active_collectors=$($snapshot.ActiveCollectors.Count) " +
        "orphan_collectors=$($orphanCollectors.Count)"
    )
    if (
        $feederRoots -ne 1 -or
        $collectorRoots -ne 1 -or
        $snapshot.ActiveCollectors.Count -gt 1 -or
        $orphanCollectors.Count -ne 0
    ) {
        throw "campaign loop verification failed"
    }
    exit 0
}

# A feeder has no valuable in-flight simulation work, so stop its tree directly.
$feederIds = @()
foreach ($target in $snapshot.Feeders) {
    $feederIds += Get-DescendantIds $target.ProcessId $snapshot.All
}
$feederIds = @($feederIds | Sort-Object -Unique -Descending)
foreach ($processId in $feederIds) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

# Stop only the bash loop layers. An active collect_wave Python process is allowed
# to finish its parquet transaction before a replacement loop starts.
foreach ($collector in $snapshot.Collectors) {
    Stop-Process -Id $collector.ProcessId -Force -ErrorAction SilentlyContinue
}

$deadline = (Get-Date).AddSeconds($CollectorWaitSeconds)
do {
    Start-Sleep -Seconds 2
    $current = Get-CampaignProcesses
    $activeCollectors = @($current.ActiveCollectors)
    if (-not $activeCollectors.Count) {
        break
    }
    if ((Get-Date) -ge $deadline) {
        $ids = @($activeCollectors | Select-Object -ExpandProperty ProcessId)
        throw "collector did not finish within ${CollectorWaitSeconds}s: $($ids -join ',')"
    }
} while ($true)

Start-Sleep -Seconds 1
$survivors = Get-CampaignProcesses
if ($survivors.Feeders.Count -or $survivors.Collectors.Count -or $survivors.ActiveCollectors.Count) {
    $ids = @(
        $survivors.Feeders + $survivors.Collectors + $survivors.ActiveCollectors |
        Select-Object -ExpandProperty ProcessId
    )
    throw "campaign loop processes survived stop: $($ids -join ',')"
}
Write-Output "campaign loops stopped and verified"
