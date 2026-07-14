# N=3 AEDT-sharing canary

These scripts only submit when the supervisor explicitly runs `submit_host.py`
or `fill_and_submit_clients.py`. Run them from the repository root in
PowerShell. Replace the allocation and account values first.

```powershell
$ROOT = 'Y:/git/MFT_pooled_refill_controller_260713'
$KIT = "$ROOT/regression_260707/campaign/n3_canary"
$PY = 'C:/Users/peets/anaconda3/envs/pyaedt2026v1/python.exe'
$STATE = 'C:/Users/peets/slurm_scheduler_runtime/mft_controller/restart_v3_8_controller_state.json'
$ALLOCATION_ID = 12345
$ACCOUNT = 'replace-me'
Set-Location $ROOT

& $PY "$KIT/make_payloads.py" --allocation-id $ALLOCATION_ID --account $ACCOUNT --state $STATE --out "$KIT/payloads"
$hostReply = & $PY "$KIT/submit_host.py" --payload "$KIT/payloads/host_payload.json" | ConvertFrom-Json
$HOST = [int]$hostReply.host_task_id

$discoveryDeadline = (Get-Date).AddMinutes(35)
while ($true) {
    $hostTask = Invoke-RestMethod "http://127.0.0.1:8000/api/tasks/$HOST"
    if ($hostTask.status -in @('completed', 'failed', 'cancelled', 'timeout')) {
        throw "host became terminal before discovery: $($hostTask.status)"
    }
    try {
        $stdout = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8000/api/tasks/$HOST/stdout?max_bytes=1048576").Content
    } catch {
        $stdout = ''
    }
    if ($stdout -match '(?m)^NODE_CANARY_DISCOVERY ') { break }
    if ((Get-Date) -ge $discoveryDeadline) { throw '35-minute discovery deadline elapsed' }
    Start-Sleep -Seconds 10
}

$clientReply = $stdout | & $PY "$KIT/fill_and_submit_clients.py" --host-task-id $HOST --payload-dir "$KIT/payloads" | ConvertFrom-Json
$CLIENTS = @($clientReply.client_task_ids)
& $PY "$KIT/verify_n3.py" --host $HOST --clients ($CLIENTS -join ',')
```

The GET of host stdout is the preferred discovery path and matches the
controller. The pinned host contract places the live file at
`/tmp/mft-aedt-n3canary-260714.discovery.json` on the compute node and removes
it when the host exits. The host clone root
`~/slurm_scheduler/runs/mft-aedt-n3canary-260714-host` is on GPFS, but the
discovery file itself is not. If direct retrieval is needed while the host is
running, first GET the host task and copy its `slurm_job_id` and
`actual_node_name`, then use the scheduler SSH helper (its own virtualenv is
required for Paramiko):

```powershell
$hostTask = Invoke-RestMethod "http://127.0.0.1:8000/api/tasks/$HOST"
$SLURM_JOB_ID = $hostTask.slurm_job_id
$NODE = $hostTask.actual_node_name
Y:/git/slurm_scheduler/.venv/Scripts/python.exe Y:/git/slurm_scheduler/scripts/run_remote.py --accounts Y:/git/slurm_scheduler/config/accounts.yaml --account $ACCOUNT "srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --ntasks=1 --nodelist=$NODE cat /tmp/mft-aedt-n3canary-260714.discovery.json"
```

That raw JSON can be piped to `fill_and_submit_clients.py`, or saved and passed
with `--discovery-json <path>`. The option also accepts literal JSON. The script
always uses the controller's deterministic GPFS host clone root.

To abort, cancel every client first and the host last. These POSTs are for the
supervisor only:

```powershell
$CLIENTS | ForEach-Object {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/tasks/$_/cancel?expected_statuses=queued,attaching,running"
}
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/tasks/$HOST/cancel?expected_statuses=queued,attaching,running"
```

If a client POST fails partway through, the script prints compact error JSON
containing `submitted_client_task_ids` to stderr. Copy those IDs into `$CLIENTS`
and run the same client-first abort sequence; do not resubmit blindly.
