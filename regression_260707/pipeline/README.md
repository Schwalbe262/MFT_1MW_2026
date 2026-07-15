# Continuous generation pipeline

This package adds a durable control plane around the existing MFT collector,
checkpoint trainer, Optuna tuner, NSGA-II optimizer, and verification tools.
It does not replace the physics/quality contracts and does not contact Slurm
until an explicitly configured verification command does so.

## Guarantees

- Dataset, tuning, optimization, and verification hand-offs are immutable,
  SHA-256 addressed directories. `manifest.json` authenticates every file and
  `COMPLETED` is installed last.
- `jobs.sqlite3` stores the job type, idempotency key, input/output generation,
  dependency DAG, owner lease, heartbeat, attempt, retry time, and terminal
  reason. Expired ownership is recovered without two workers completing the
  same lease.
- Collector, trainer, tuner, optimizer, standard verifier, and fine verifier
  are independent supervised lanes. A newer dataset can be collected/trained
  while NSGA-II or FEA is still consuming an older immutable model generation.
- The checkpoint orchestrator is the only model promoter. Active learning only
  adopts its accepted `current.json` generation.
- Production policy is fixed at model activation >=3,000 strict-full rows,
  first tuning >=4,000, later tuning after `max(2,000 rows, 20%)` growth or an
  explicit drift/quality signal, 16 NSGA restarts at population 200 using no
  more than four processes, standard top 33, and fine top 3.

## Operation

Use the deployed PyAEDT environment and explicit full revision SHAs:

```powershell
$py = 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe'
$solver = '<40-character solver SHA>'
$library = '<40-character pyaedt_library SHA>'

& $py -m regression_260707.pipeline `
  --runtime-root regression_260707 control `
  --solver-revision $solver --library-revision $library --once

& $py -m regression_260707.pipeline `
  --runtime-root regression_260707 supervise
```

Run `control` without `--once` for recurring generation discovery. Run
`status` to inspect queue ownership, retries, and terminal reasons. The
controller and supervisor are separate processes so long model tuning cannot
delay collection.

Verification is fail-closed by default: no standard/fine job is enqueued until
`--verification-commands` explicitly enables the reviewed scheduler adapter.
When absent, the controller result reports `verification_standard` and
`verification_fine` as blocked; it never reports them complete. The JSON file
has this sealed form (use the same reviewed library checkout for both stages):

```json
{
  "standard": {
    "adapter": "mft_scheduler_v1",
    "execute": true,
    "library_root": "Y:/git/pyaedt_library"
  },
  "fine": {
    "adapter": "mft_scheduler_v1",
    "execute": true,
    "library_root": "Y:/git/pyaedt_library"
  }
}
```

The adapter authenticates the optimization generation, deterministically
selects exactly 33 Pareto-spanning candidates, and uses the existing hardened
`verify/scheduler_client.py` submission/reconciliation path with pinned
solver/library revisions and the reviewed standard profile. Fine verification
accepts only the three smallest actual-volume standard passes, then uses the
reviewed fine profile. Both stages require an exact terminal result inventory;
unknown task identities, missing results, weakened counts, and mismatched
candidate parameters fail closed.

`training/tune_optuna.py` publishes `tuning/<sha>/params.json`; the trainer DAG
passes that exact file with `--params`, and `train_report.json` records its
path and SHA. The legacy mutable `training/best_params.json` is written only
when `--legacy-output` is explicitly requested.
