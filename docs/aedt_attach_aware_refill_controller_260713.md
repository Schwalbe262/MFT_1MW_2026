# Attach-aware MFT 500-maintenance controller (restart v3, not started)

## Outcome

This branch provides the reviewed restart-v3 500-maintenance controller for a
rolling standalone-to-pooled AEDT transition. PLAN mode performs scheduler
GETs only. The controller has not been started by this change, and no task was
submitted or cancelled while preparing it.

The logical unit remains one simulation project and one expected training row.
The scheduler continues to count each simulation as one MFT project task, so
the project concurrency target remains exactly 500. A pooled bundle groups up
to `projects_per_aedt=N` project tasks and records
`bundle.expected_rows == project count`; it does not reinterpret the scheduler
task cap as a Desktop cap.

## Modes and invariants

- `standalone + fea_bursty`: existing one-project/one-Desktop execution.
- `pooled + fea_bursty`: each logical project is a normal MFT task with
  `aedt_backend=pooled`; scheduler pool leases place up to N projects on an AEDT.
- Both scheduling modes retain `fea_bursty`. The AEDT ownership selector is
  independent from CPU/memory overcommit policy.
- Production currently pins `pooled_fraction=0.0`, so every replacement is
  standalone until the N=2 canary is separately accepted. The N=2 admission
  and capacity plumbing remains present but inactive.
- `desired_aedt_sessions = ceil(pooled logical projects / N)` and may never
  exceed the UI/operator `max_aedt_sessions` ceiling.
- A pooled policy is invalid if `max_aedt_sessions * N < 500`.
- N is not hard-coded to two. The requested N must be no larger than the
  separately pinned `validated_projects_per_aedt`. Moving to N=3 therefore
  requires N=3 evidence and a new immutable policy, not a code rewrite.
- No controller path has mass-cancel authority. Existing standalone and
  already-running pooled tasks drain naturally.

## Fast rolling application

1. Deploy the exact attach-capable MFT and scheduler revisions recorded in the
   policy provenance.
2. In scheduler Web UI set the AEDT session ceiling, logical project target,
   and `projects_per_aedt`. For the current N=2 policy, target 500 needs 250
   sessions, exactly matching the prepared ceiling.
3. Keep the production pooled fraction at zero. Enable pooled admission only
   after the N=2 scheduler validation row is accepted and the pool reports
   `enabled`, `validation_passed`, and `operational`.
4. Replace the controller process at an immutable terminal cycle boundary.
   Do not run old and new controllers together.
5. Let existing tasks drain. With the current zero fraction, every refill is a
   standalone restart-v3 task; there is no mass cancellation or generation
   cutover gap.
6. After a separately reviewed nonzero pooled policy is installed, a missing
   or mismatched pool gate still falls back to the unchanged standalone path.

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

Every submitted params payload explicitly includes
`physics_data_revision=mft1mw-1k101-native-lamination-kf0p85-v3` and
`core_lamination_factor=0.85`. Both values are also part of the provenance and
generation digests used to scope scheduler dedupe. The collector still decides
whether rows are admitted to training; pooled transport does not weaken that
quality gate.

Any future change to the 1K101 interpretation requires a copied policy with a
new `solver_revision`, `physics_data_revision`, data contract and generation.
The controller then fills only newly opened slots; cancellation of older work,
if separately authorized, remains outside this controller.

## Read-only plan command

`plan` reads the live project and active task inventories, initializes only the
fresh ignored `restart_v3_controller_state.json` when absent, and emits the
exact rolling-replacement actions. The state begins at cursor 127, after the
35 valid candidates reserved/used by the restart pilot, and new names retain
the collector-compatible `mft-camp-` prefix. PLAN contains no scheduler
mutation call.

```powershell
& 'C:/Users/peets/anaconda3/envs/pyaedt2026v1/python.exe' -u `
  regression_260707/campaign/attach_aware_refill_controller.py plan `
  --policy regression_260707/campaign/attach_refill_policy_canary_n2.json `
  --scheduler-url http://127.0.0.1:8000 `
  --state-path regression_260707/campaign/restart_v3_controller_state.json
```

`run` requires the exact generation ID emitted by PLAN and holds the common
campaign mutation lock. It submits only the live logical deficit and advances
the fresh state after each idempotently accepted/reconciled task. There is no
mass-cancel path.

## Exact staged revisions

- Production solver: `06c650cfa0c1be2bcd8af9a8f074fe8fae701d0d`
- Pyaedt library: `e6b9b9d20a832ff5c3f7ca97218737a0b8650781`
- Physics data: `mft1mw-1k101-native-lamination-kf0p85-v3`
- Core lamination factor: `0.85`

- Mature controller base: `5f3987e00212ffebe88d4d705b07c8f52603c5ed`
- Selector contract base: `9f7b005927bb4d039c743623e8f39e9a214feca0`
- Current bounded scheduler runtime: `d66cf542b0deffa465d43367dd89ed8a4639235d`
- Current MFT attach canary: `78d95110505315b8852c286994697844e22c214a`
- Validated attach implementation: `fd3b02c2a4c3bd2ef566cc6e79ce1291f5576a18`
- Normal/pre-solve validation scheduler: `620a3019da3577bbe2ea2e10aafebd8c26717df7`
- Active-timeout validation scheduler: `f3a65eeabb0e1d4d0f7694035147b9c27e673ee1`
