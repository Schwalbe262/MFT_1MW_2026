# Production release gate (8010)

Do not replace the running 8010 monitor merely because dependencies install
or unit tests pass.  The candidate checkout and virtual environment are an
immutable pair.  A release is deployable only when `release_canary.py` emits a
deployment manifest with `deploy_authorized: true`.

The candidate environment must use PyArrow 24.x.  PyArrow 19 can inspect the
canonical Parquet schema but fails while reading its data with `Repetition
level histogram size mismatch`; a manifest-only check is therefore not an
acceptable substitute for the read canary.

Example (run from the clean candidate checkout, before stopping 8010):

```powershell
$revision = git rev-parse HEAD
$source = (Get-Location).Path
$venv = 'C:\Users\peets\slurm_scheduler_runtime\mft_monitor_releases\<revision>\venv'

& "$venv\Scripts\python.exe" -m regression_260707.monitoring.release_canary `
  --source-root $source `
  --regression-root "$source\regression_260707" `
  --pipeline-root 'C:\Users\peets\slurm_scheduler_runtime\mft_pipeline' `
  --rolling-index 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_nsga_slurm_rolling\canonical\index.json' `
  --current7-index '<absolute-current7-canonical-index.json>' `
  --current7-condition-index '<absolute-entry-condition-index.json>' `
  --current7-condition-index '<absolute-bridge-condition-index.json>' `
  --current7-condition-index '<absolute-close-condition-index.json>' `
  --current7-condition-index '<absolute-final-condition-index.json>' `
  --blocker-hpo-v2-status '<absolute-blocker-hpo-v2-status.json>' `
  --evidence-dir 'C:\Users\peets\slurm_scheduler_runtime\mft_monitor_release_evidence' `
  --expected-revision $revision `
  --expected-venv-root $venv
```

The canary is read-only with respect to campaign, scheduler, AEDT, model and
NSGA state.  It disables monitor history, starts the candidate ASGI app only
on an ephemeral loopback port, and verifies:

- the exact clean source commit and monitoring-tree content hash;
- the exact Python executable/venv and PyArrow 24.x package identity;
- on Windows, the exact `pyvenv.cfg` home/executable relationship plus the
  SHA-256 identities of both the venv launcher and its observed base process
  image (no unrelated base interpreter is accepted after the switch);
- the authenticated controller dataset generation and its Parquet SHA-256;
- a real full-validation read of the primary provenance, design, matrix,
  loss, flux, thermal, capacitance and resonance columns;
- `/api/data` authenticated strict cohort counts;
- `/api/nsga2` rolling index integrity, model/constraint identity, terminal
  results, active/queued counters, Pareto count and near-feasible count;
- when `--current7-index` is present, the regular non-reparse secondary index,
  its status and compatibility references (including their declared SHA-256),
  and `/api/nsga2.tier1_current7_search` with the exact T110/res15k/current7
  constraint identity and all execution-authority flags disabled;
- every repeated `--current7-condition-index`, in the declared order, including
  its regular non-reparse pointer, containment, nested SHA-256 identities and a
  real bounded status read up to 64 MiB; the matching condition projection must
  be integrity-verified, read-only, non-authoritative and GUI-launch-disabled;
- when `--blocker-hpo-v2-status` is present, the regular non-reparse status
  snapshot SHA-256, sealed per-target job/trial accounting,
  disabled promotion/FEA authority, and the matching
  `/api/dashboard.continuous_pipeline.blocker_hpo_v2` projection;
- `/api/nsga2/progress` as a summary of that same authoritative contract;
- two consecutive fresh reads (up to 120 seconds, every 2 seconds) in which
  `/api/nsga2` and `/api/nsga2/progress` expose the exact same rolling
  generation; stale status remains unavailable and a timeout fails closed;
- `/api/local-aedt-gui/launches`, including the immutable standalone solver
  revision, runner hash, result schema and retained-session capacity.

A failed run writes canary evidence with `deploy_authorized: false` but does
not write a deployment manifest.  A passing run writes both the evidence and
an immutable, timestamped deployment manifest plus its SHA-256 sidecar.  Keep
the old 8010 process running until those artifacts have been reviewed.  This
procedure never targets the scheduler UI/service on 8002.

The primary current7, repeated condition indexes, and HPO status are independent
optional monitor lanes.  If an
argument is omitted, the canary removes that variable from the candidate
ephemeral process and preserves legacy behavior.  If supplied, the deployment
manifest carries the absolute path and observed content SHA-256.  Since the
producer replaces current7 indexes and the HPO heartbeat atomically while they
run, post-transition verification requires the same sealed paths in the same
order, authenticates the new content and records its new SHA-256; it does not
incorrectly require a live mutable file to retain its pre-canary byte hash.

When condition indexes are supplied without `--current7-index`, the canary uses
condition-archive mode.  It requires two consecutive coherent reads of all
condition generations and refuses any execution authority, while allowing the
legacy primary search to remain unavailable.  This mode is intended for the
four staged final-goal views after the protected primary current7 producer has
stopped; it does not relabel a stale primary snapshot as live.

Immediately before replacing 8010, seal its unique listener PID, exact command,
executable hash, working directory, clean source revision and rollback venv:

```powershell
& "$venv\Scripts\python.exe" -m regression_260707.monitoring.runtime_transition seal `
  --rollback-source-root '<current-8010-source-root>' `
  --rollback-venv-root '<current-8010-venv-root>' `
  --deployment-manifest '<passing-deployment-manifest.json>' `
  --evidence-dir 'C:\Users\peets\slurm_scheduler_runtime\mft_monitor_release_evidence'
```

The command is inspection-only.  Its output contains the exact rollback
command; keep the resulting pre-seal and SHA sidecar together.  After the 8010
process has been replaced, verify that the listener PID and executable changed
to the candidate identity and that the served browser bundle really polls the
new endpoints:

```powershell
& "$venv\Scripts\python.exe" -m regression_260707.monitoring.runtime_transition verify `
  --pre-seal '<runtime_pre_seal_*.json>' `
  --deployment-manifest '<passing-deployment-manifest.json>' `
  --evidence-dir 'C:\Users\peets\slurm_scheduler_runtime\mft_monitor_release_evidence'
```

If post-verification fails, use the pre-sealed command/cwd/venv to restore the
old 8010 runtime.  Neither transition command stops, starts, or modifies a
process; process control remains an explicit operator action.

Post-transition verification also reads the two optional environment variables
from the unique listener process.  An undeclared variable, a path mismatch, a
reparse point, a nested SHA mismatch, or a failed current7/HPO API projection
fails closed and produces no passing post-transition seal.
