# Final1000 multi-seed release and 500-lane runbook

Status: local release candidate. No bundle publication, Scheduler submission,
controller stop, remote write, AEDT, or FEA action has been performed.

## Resource objective

The operational target is exactly 500 active physical standard Scheduler lanes
with stage quotas 200/160/90/50. Each lane owns four ordered logical seeds and
runs one fresh optimizer subprocess at a time with 4 Scheduler CPUs,
4 inference threads, and 1 optimizer process. This keeps the allocated SLURM
CPU capacity busy while reducing attach/refill churn; it does not claim that
four children inside one lane run simultaneously.

The controller and monitoring surfaces always report physical active lanes and
logical seed work separately. A normal refill must stop at physical 500 even
though 500 four-child lanes can own up to 2,000 logical seeds.

## Immutable release preparation

`tools/tier1_final1000_multiseed_release.py` accepts only a clean checkout whose
history contains science base `579c651`. For the four-stage form it also
authenticates the immutable 30811ea release gate, all four parent plan hashes,
and the pass-2 publication receipts.

The release closure currently contains 21 repository modules. In addition to
the lane execution roots it includes the production consumer and its complete
harvest/status/monitor/controller transitive closure:

- `tools/tier1_corrected_current7_receipt.py`
- `tools/tier1_corrected_current7_slurm_bundle.py`
- `tools/tier1_corrected_current7_slurm_controller.py`
- `tools/tier1_corrected_current7_slurm_harvest.py`
- `tools/tier1_corrected_current7_slurm_publish.py`
- `tools/tier1_corrected_current7_slurm_seed_runner.py`
- `tools/tier1_corrected_generation_adapter.py`
- `tools/tier1_corrected_generation_preflight.py`
- `tools/tier1_deep_crossover_contract.py`
- `tools/tier1_final1000_multiseed_contract.py`
- `tools/tier1_final1000_multiseed_consumer.py`
- `tools/tier1_final1000_multiseed_controller.py`
- `tools/tier1_final1000_multiseed_harvest.py`
- `tools/tier1_final1000_multiseed_lane_runner.py`
- `tools/tier1_final1000_multiseed_monitor.py`
- `tools/tier1_final1000_multiseed_status.py`
- `tools/tier1_final1000_rolling_migration.py`
- `tools/tier1_final1000_slurm_controller.py`
- `tools/tier1_final1000_slurm_harvest.py`
- `tools/tier1_final1000_slurm_launch.py`
- `tools/tier1_final1000_stage_profiles.py`

The tool overlays the complete computed closure and every parent-bundled
repository source that still exists, replaces `.source-revision`, performs an
isolated `python -I` import smoke, and seals deterministic candidate/delta
plans. READY is never copied, generated, or reused. Preparation itself has no
remote transport and no Scheduler client.

Do not submit tasks until a separate release authorization is given. Bundle
publication is a distinct, explicit operation: the generic delta CLI is not a
valid publication surface for this release because its production guard is
pinned to the older single Current7 parent.

### Authenticated publication commands

The multi-seed publisher authenticates the sealed four-stage preparation,
the exact 30811ea remote gate and its file/evidence hashes, all four stage
receipt files, the clean checkout revision, canonical contained candidate and
delta paths, each exact delta-plan receipt, and each stage's dynamically
derived parent identity. It always passes that dynamic identity to the
hardened delta publisher. It never calls the delta publisher with an
unconstrained parent.

The prepared 77a9d485 release can be revalidated with four remote-write-free
dry runs using these exact PowerShell commands. No Scheduler transport is
constructed in dry-run mode:

```powershell
$Preparation = 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\multiseed_release_77a9d485_260722\release_preparation_receipt.json'
$Revision = '77a9d485fd81b5405b837a04cc003f0e37e13a32'
$Stages = @('entry-1200-t125', 'bridge-1150-t115', 'close-1075-t107p5', 'final-1000-t100')
foreach ($Stage in $Stages) {
    python tools\tier1_final1000_multiseed_publish.py --preparation-receipt $Preparation --stage $Stage --expected-checkout-revision $Revision
}
```

After an independent release authorization, publication itself requires both
`--apply` and a new explicit receipt path. An existing receipt path is refused.
This loop publishes exactly those four prepared deltas and performs no task
submission, controller stop, AEDT, or FEA action:

```powershell
$ReceiptRoot = 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\multiseed_release_77a9d485_260722\publication_receipts'
foreach ($Stage in $Stages) {
    python tools\tier1_final1000_multiseed_publish.py --preparation-receipt $Preparation --stage $Stage --expected-checkout-revision $Revision --apply --receipt-out "$ReceiptRoot\$Stage.json"
}
```

The apply path uses the same bounded/batched `SSHDeltaTransport` and Scheduler
account loader as the hardened delta publisher. READY is written only by that
publisher after complete remote authentication; no READY file may exist in
the local preparation tree.

## Required rollout gates

1. Seal and revalidate the exact 8010 adapter code, backend code, compact index,
   and test identities. No canary POST is allowed without this capability.
2. Keep the old v1 controller progressing at physical 500. Submit exactly four
   additive batch-1 lanes, one per stage, using reserved seed `end-5`. The
   temporary physical ceiling is 504. All four must terminally authenticate as
   passed.
3. Submit exactly four additive batch-4 lanes, one per stage, using reserved
   seeds `end-4..end-1`. Again the ceiling is 504 and all four must pass. There
   is no ordinary refill during either gate.
4. Wait for the predecessor controller's final stopped state to stabilize.
   Import its final SHA, revision, entry count, four exact harvest cohorts, and
   the cohort-specific 200/300-generation task envelopes. The initial shadow
   snapshot is never accepted for cutover.
5. Persist `cutover_prepared` plus the exact upgraded controller export before
   starting canonical consumption. This phase is controller-write-only and
   performs refill POST0. Artifact-first/state-second persistence makes a
   crash restart deterministic without an operator-authored JSON extraction.
6. Require capability schema v2, the exact four-index v1 harvester handoff,
   old PID exit, canonical runtime containment, and an active OS writer lease.
   The receipt must bind the prepared controller SHA before refill is released.
7. Release cancellation-free natural replacement to exactly 500 physical
   lanes and quotas 200/160/90/50. The driver uses only `POST /api/tasks`; it
   has no cancellation or preemption path.

## Driver behavior

The driver is POST0/write0 by default. `--watch` is rejected unless `--apply`
is also explicit. Every apply cycle revalidates monitoring capability, reads a
latest-10,000 bulk inventory with missing-detail fallback, reconciles exact
name/dedupe/task envelopes, and atomically advances a content-addressed state
history.

A transient GET receives bounded exponential backoff. A POST is never blindly
retried because its result can be ambiguous. If a process dies after POST but
before state persistence, restart derives the same gate/refill task and seed,
finds its exact dedupe in latest10k, recovers the Scheduler task ID, and issues
zero additional POSTs. A terminal gate failure, capability drift, state-chain
tamper, missing active task, or envelope mismatch fails closed.

Every refill reconcile also requires a fresh canonical capability for its
current exported controller SHA and an active writer lease. Missing capability
or a healthy receipt still describing C(n-1) performs reserve/POST0 while the
15-second consumer catches the 30-second driver. Stale, shadow, tampered, or
unlocked receipts are fatal. This prevents continued refill if UI harvesting
has stopped.

The operator stop is a sealed latch checked before the next Scheduler GET. It
does not cancel already submitted work.

## Local validation snapshot

The exact live read-only probe captured predecessor revision 906 with 9,497
entries and active physical count 500. Upgrade preserved four cohorts and
their generation envelopes:

- predecessor `3835df...`: 2,703 entries at 200 generations;
- successor `d1450...`: 2,361 entries at 200 generations;
- successor `f2b0b...`: 638 entries at 300 generations;
- current successor `3f7b9...`: 3,795 entries at 300 generations.

The source mapping remained unchanged in memory, JSON restart validation was
identity-preserving, and the probe performed zero Scheduler HTTP requests,
remote reads/writes, or source writes. Exact file and state hashes are sealed
in `tests/fixtures/tier1_final1000_live_v3_chain.json` and the accompanying
local evidence JSON.
