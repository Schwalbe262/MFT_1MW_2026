# Attach-aware MFT 300-maintenance controller (prepared, not deployed)

## Outcome

This branch prepares the reviewed 300-maintenance controller for a rolling
standalone-to-pooled AEDT transition. It does not start a second controller,
change the live pool, submit a task, or cancel a live task.

The logical unit remains one simulation project and one expected training row.
The scheduler continues to count each simulation as one MFT project task, so
the project concurrency target remains exactly 300. A pooled bundle groups up
to `projects_per_aedt=N` project tasks and records
`bundle.expected_rows == project count`; it does not reinterpret the scheduler
task cap as a Desktop cap.

## Modes and invariants

- `standalone + fea_bursty`: existing one-project/one-Desktop execution.
- `pooled + fea_bursty`: each logical project is a normal MFT task with
  `aedt_backend=pooled`; scheduler pool leases place up to N projects on an AEDT.
- Both scheduling modes retain `fea_bursty`. The AEDT ownership selector is
  independent from CPU/memory overcommit policy.
- `desired_aedt_sessions = ceil(pooled logical projects / N)` and may never
  exceed the UI/operator `max_aedt_sessions` ceiling.
- A pooled policy is invalid if `max_aedt_sessions * N < 300`.
- N is not hard-coded to two. The requested N must be no larger than the
  separately pinned `validated_projects_per_aedt`. Moving to N=3 therefore
  requires N=3 evidence and a new immutable policy, not a code rewrite.
- No controller path has mass-cancel authority. Existing standalone and
  already-running pooled tasks drain naturally.

## Fast rolling application

1. Deploy the exact attach-capable MFT and scheduler revisions recorded in the
   policy provenance.
2. In scheduler Web UI set the AEDT session ceiling, logical project target,
   and `projects_per_aedt`. For the current N=2 policy, target 300 needs 150
   sessions; the prepared ceiling is 250.
3. Enable the pool only after its scheduler validation row is passed and the
   pool reports `enabled`, `validation_passed`, and `operational`.
4. Replace the controller process at an immutable terminal cycle boundary.
   Do not run old and new controllers together.
5. Let the existing 300 tasks drain. Each refill deficit is submitted through
   pooled attach, so production validation begins immediately without a mass
   cancellation or a temporary drop below the 300 logical target.
6. If the pool gate is unavailable or mismatched, that refill cycle uses the
   unchanged standalone backend. This avoids a refill outage while retaining
   the exact same `fea_bursty` placement policy.

## Failed pooled bundle rollback

The bundle ledger waits until every sibling is terminal. Collected valid rows
remain accepted. Only missing or invalid rows receive new standalone task
identities. The decision always has `cancel_task_ids=[]` and cannot affect
another bundle. The scheduler pool remains responsible for quarantining and
recycling the affected Desktop; the controller does not kill a shared Desktop.

## Revision and data provenance

Every task gets a provenance-scoped dedupe identity and these environment
records in stdout:

- controller policy SHA-256;
- solver and pyaedt-library revisions;
- data-contract revision;
- scheduler selector/runtime revisions;
- attach canary implementation revision;
- normal/abort and active-timeout validation revisions;
- bundle ID, bundle expected rows, per-project index, and N.

Changing the fill-factor/1K101 interpretation must create a new
`data_contract_revision` and solver revision. The scoped dedupe prevents the
new physical interpretation from reconciling to an old task. The collector
still decides whether rows are admitted to training; pooled transport does not
weaken that quality gate.

When the 1K101 result is approved, copy the policy file, update
`solver_revision`, `data_contract_revision`, and any validation SHA, then
replace only the refill generation at a terminal controller cycle. The new
controller fills newly opened slots with the new generation; cancellation of
old simulations, if separately authorized, is outside this controller.

## Read-only plan command

`attach_aware_refill_controller.py` is a read-only planner. It consumes an
immutable policy, candidate identities, a scheduler `/api/aedt-pool` snapshot,
and the current logical MFT active count. It emits bundle and per-task options
and cannot mutate the scheduler.

```powershell
python regression_260707/campaign/attach_aware_refill_controller.py `
  --policy regression_260707/campaign/attach_refill_policy_canary_n2.json `
  --candidates prepared_candidates.json `
  --pool-status aedt_pool_status.json `
  --active-project-tasks 294 `
  --output prepared_refill_plan.json
```

The reviewed mature controller must consume the plan inside its existing
campaign mutation lock and immutable cycle journal. The scheduler submission
client now accepts `aedt_backend`, `scheduling_profile`, provenance environment
and a provenance dedupe scope; all defaults remain standalone/`fea_bursty`.

## Exact staged revisions

- Mature controller base: `5f3987e00212ffebe88d4d705b07c8f52603c5ed`
- Selector contract base: `9f7b005927bb4d039c743623e8f39e9a214feca0`
- Current bounded scheduler runtime: `d66cf542b0deffa465d43367dd89ed8a4639235d`
- Current MFT attach canary: `78d95110505315b8852c286994697844e22c214a`
- Validated attach implementation: `fd3b02c2a4c3bd2ef566cc6e79ce1291f5576a18`
- Normal/pre-solve validation scheduler: `620a3019da3577bbe2ea2e10aafebd8c26717df7`
- Active-timeout validation scheduler: `f3a65eeabb0e1d4d0f7694035147b9c27e673ee1`
