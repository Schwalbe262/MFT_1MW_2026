# Blocker HPO v2 stale-run recovery

Blocker HPO v2 deliberately does not auto-steal an expired lease and does not
rewrite an Optuna `RUNNING` trial. A hard process termination can occur after a
model fit has started but before its objective evidence is committed. Treating
that trial as complete, waiting, or retryable without an audit would mix
unknown computation into the production parameter search.

## Normal restart

If the previous invocation exited normally, it removes
`.mft-blocker-hpo-v2.lock`. Every study must then contain only authenticated
`COMPLETE` trials followed by authenticated `WAITING` deterministic proposals.
The next configured cumulative stage can resume those studies. Study, objective,
implementation, feature, target, search-space, runtime-package, holdout, and
proposal fingerprints must all match exactly.

## Hard-exit indication

Either condition is a fail-closed stale run:

- the v2 lock file remains, even when its `expires_epoch` is in the past;
- a v2 SQLite study contains `RUNNING`, `FAIL`, or `PRUNED` state.

The lease expiry is diagnostic evidence only. It is not permission to delete
the lock or reuse the databases.

## Authenticated recovery

1. Stop the dedicated Task Scheduler task and disable its automatic restart.
2. Verify that no process command line contains `tune_blockers_v2.py` and that
   the lock's PID is absent on the recorded host.
3. Preserve the complete study root, including the lock and every SQLite side
   file, as an immutable quarantine copy. Record a recursive file-name, size,
   and SHA-256 inventory next to that copy.
4. Do not edit, delete, mark-failed, or resume any database in the quarantined
   root.
5. Select a brand-new empty fixed-local-drive study root. Do not place it under
   the exploratory v1 root.
6. Run `--preflight-only` again and compare its dataset, quality, implementation,
   feature, runtime, and outer-holdout fingerprints with the intended v2
   contract.
7. Start the configured v2 stage against the new root. Deterministic proposals
   restart from ordinal zero, so the replay is auditable and does not import the
   stale trial or the exploratory v1 stage.

This recovery recomputes completed work by design. Preserving trustworthy
production acceptance evidence is more important than reusing an ambiguous
in-flight model fit.

## Task Scheduler policy

Use a dedicated one-shot task with a durable stdout/stderr path and external
heartbeat monitoring. A restart wrapper may retry only after a normal exit.
When a lock remains or the process disappears without a result receipt, the
wrapper must stop and raise an operator-visible stale-run alert; it must never
remove the lock automatically.

The production invocation also supplies a local `--status-json`.  The task
must run with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` set to `1`, `--model-threads 1`, `--job-workers 8`,
`--max-model-thread-budget 8`, and `--max-rss-gb 64`.  Task Scheduler owns the
foreground wrapper and Python child, has no execution time limit or automatic
restart, and uses an empty fixed-drive study root after any stale-run event.
