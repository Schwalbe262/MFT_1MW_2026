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

Do not run preparation against the real release root, publish its output, or
submit its tasks until a separate release authorization is given.

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
