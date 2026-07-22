# Tier-1 final1000 multi-seed Phase-B rollout gates

Status: implementation, tests, and immutable dry evidence only. No Scheduler
task was submitted/cancelled, no bundle was deployed/published, Scheduler 8002
was not changed, and no FEA/AEDT action was performed. Phase B is deliberately
`production_eligible=false` until the consumer and authority gates below are
independently released.

## Concurrent execution contract

- One logical seed remains the canonical Phase-A child task: a fresh process,
  independent RNG/model load, immutable payload/dedupe identity, seed-local
  journal/receipt, 4 CPUs, and 28,672 MiB. There is no model or RNG reuse.
- One physical Phase-B parent runs exactly 1..8 children concurrently and asks
  for `4 * lanes` CPUs and `28,672 * lanes` MiB. The first canary is 4 children
  (16 CPUs/114,688 MiB); the normal full parent is 8 children
  (32 CPUs/229,376 MiB). The 1..7 shapes are exact capacity-tail parents.
- The outer Scheduler step already owns its CPUs through
  `srun --exact --exclusive`. A nested exclusive `srun -c4` can wait on CPUs
  already owned by the outer step, so it is prohibited. The parent partitions
  its inherited Linux affinity mask; each child applies an exact disjoint
  four-CPU `sched_setaffinity` mask before model/optimizer creation.
- Model loads are staggered 5--10 seconds (default 5 seconds). The lower
  reviewed bound fills an eight-child parent in 35 seconds instead of 52.5
  seconds without allowing a model-load start burst. Children may
  finish out of order, but receipts are published only as a durable contiguous
  ordinal prefix. Physical Scheduler task IDs are preserved; no virtual task ID
  is invented.
- A seed-local optimizer failure is sealed without stopping healthy siblings.
  Model-load/preflight identity failure is parent-fatal. SIGTERM/SIGINT/deadline
  terminates all active process groups, observes a kill grace, and seals the
  prefix. An existing parent journal permanently refuses same-ID restart;
  recovery requires a new physical parent.

### Scratch isolation is not the ENOSPC fix

Each child gets private `TMPDIR`, `TMP`, `TEMP`, `JOBLIB_TEMP_FOLDER`, and
`XDG_CACHE_HOME` below
`runs/task-<physical-id>/seed-<seed>/runtime`. Cleanup occurs only after that
child is reaped, never follows a replacement symlink, and never deletes shared
`/tmp` or a sibling path.

The observed ENOSPC root cause is instead repeated sklearn forest prediction:
joblib `ThreadingBackend` builds `multiprocessing.pool.ThreadPool`, whose
`SimpleQueue` creates POSIX `SemLock` objects until `/dev/shm` semaphore
capacity is exhausted. Phase B therefore additionally requires:

- sklearn forest (`extratrees`) `n_jobs=1`; native LightGBM/XGBoost/CatBoost
  may retain the bounded four-thread child budget;
- a remote preflight repeated-predict attestation over every required target,
  while joblib `ThreadPool` and `_multiprocessing.SemLock` constructors are
  fail-fast;
- zero constructor attempts, deterministic outputs, and a sealed stress hash;
- parent refusal of a nominally completed child if the legacy status lacks that
  stress hash.

The Phase-B payload and remote preflight are also bound to reviewed SemLock
release `6ea0e17e5e028ebb8d89d910c3cec1a0f014dae5`, smoke schema
`mft-tier1-semlock-safe-inference-smoke-v1`, and the exact authenticated helper
file SHA. That release serializes both ExtraTrees and RandomForest sklearn
families; a bundle without those exact helper bytes is rejected.

The focused test runs 64 repetitions over 20 targets (1,280 predict calls) with
zero ThreadPool/SemLock construction attempts. The runtime preflight performs
the same contract with a smaller fixed repeat count before optimizer release.

## Capacity-aware exact packing

Lane unit: 4 CPUs / 28,672 MiB. The reviewed current active39 empty-pool
capacity histogram is:

`{16:15, 14:1, 12:1, 11:3, 10:1, 9:2, 8:4, 7:7, 6:1, 5:3, 2:1}`

Splitting each allocation into exact parents of at most eight children yields
62 immediately placeable parents and this exact active shape:

`{8:42, 7:7, 6:2, 5:3, 4:1, 3:3, 2:2, 1:2}` = 431 logical children.

The remaining queue is exactly eight 8-child parents plus one 5-child parent:

`{8:8, 5:1}` = 69 logical children.

The exact 500-child/71-parent aggregate is therefore:

`{8:50, 7:7, 6:2, 5:4, 4:1, 3:3, 2:2, 1:2}`.

Latest logical science quotas and homogeneous parent shapes are:

| Stage | Quota | Exact parent shapes |
|---|---:|---|
| entry | 300 | `{8:26,7:7,6:1,5:4,4:1,3:3,2:1,1:2}` |
| bridge | 150 | `{8:18,6:1}` |
| close | 40 | `{8:5}` |
| final | 10 | `{8:1,2:1}` |

For the reviewed snapshot, the queued `{8:8,5:1}` tail belongs to entry; the
remaining per-stage shapes map exactly to the 431 active slots. No parent
crosses a stage, bundle, or wave. Repacking the same children into one-child
parents produces the identical logical child inventory SHA, proving that lane
count does not alter science identity.

The allocation inventory includes immutable allocation IDs and a canonical
snapshot SHA. Immediately before activation, the controller must compare a
fresh snapshot with `rendered_from_allocation_inventory_sha256`. Any change in
identity, state, or usable capacity requires rerender plus review, or a hard
stop. A generic static "8 first" pack is documentary only and cannot activate.

Scheduler ready-lane reserve is deliberately zero: the successor does not
guess where an empty node will appear or submit against predicted capacity.
Admission follows an authenticated capacity snapshot and natural-terminal
vacancies only. This avoids stranded speculative parents while the five-second
in-parent stagger bounds the only deliberate dispatch bubble. Snapshot drift
between render and activation is rerender-or-fail-closed.

Immutable dry evidence:

- `docs/evidence/tier1_final1000_multiseed_phase_b_dry_run_20260722.json`
- `docs/evidence/tier1_final1000_multiseed_phase_b_integration_20260722.json`
- `docs/evidence/tier1_final1000_multiseed_phase_b_smoke_prerelease_20260723.json`

### Immutable runtime bundle and 1/4/8 smoke release

Every production candidate must be rebuilt with the complete Phase-B runtime
closure. The bundle planner's `--phase-b-runtime` switch adds all 19 mandatory
remote and optimizer-closure modules (including the repository-root
`module/input_parameter_260706.py` dependency), while the generic code
collector also carries every tracked `module/**/*.py` file. The planner seals
their file records in `files` and `code_inventory` and adds a
`required_runtime_code` attestation. The SemLock-safe inference helper is also
bound to its reviewed hard SHA-256. Missing files, line-ending drift, or helper
content drift stops planning before any publication:

```text
python -m tools.tier1_corrected_current7_slurm_bundle plan \
  ...existing authenticated plan arguments... \
  --phase-b-runtime
```

After immutable publication, the smoke renderer accepts only an exact launch
plan, content-addressed manifest, and publisher `READY` seal. It verifies every
file and runtime identity before rendering diagnostic 1/4/8-lane envelopes.
The command prints JSON only; it has no Scheduler client and performs no POST:

```text
python -m tools.tier1_final1000_multiseed_phase_b_smoke_release \
  --launch-plan <bound-launch-plan.json> \
  --bundle-manifest <bundle_manifest.json> \
  --ready <READY.json> \
  --stage-id entry-1200-t125
```

Execution order is diagnostic `1 -> 4 -> 8`, while production promotion is
gated `4 -> 8`. The four-lane parent requests 16 CPUs / 114,688 MiB and the
eight-lane parent requests 32 CPUs / 229,376 MiB. Promotion requires zero
failures, SemLock attempts, and affinity escapes; child peak RSS at most 22
GiB; dispatch fill at most 35 seconds; bounded p50/p95 and CPU-hours/seed
regressions; and at least 1.7x eight-vs-four throughput.

The 2026-07-23 prerelease assessment is intentionally blocked. It authenticates
the real topology64 successor bytes and eight base/delta bundle manifests, but
the source successor is unbound and `launch_eligible=false`, no `READY` exists,
and those manifests contain only 9 of the 19 mandatory runtime-closure modules
with no
`required_runtime_code` attestation. Therefore it emits no smoke task and must
not be interpreted as a failed performance run. Regenerate the assessment only
after a newly integrated bundle, READY seal, and launch binding exist; never
copy task or result hashes from an older bundle.

### Exact-500 liveness reconciliation

`tier1_final1000_multiseed_phase_b_liveness.py` is a pure, bounded reconciliation
layer. A planning authority is derived deterministically from the current
durable CAS control fence. Its owner, monotonic epoch/store revision, Phase-A
handoff, watcher/submitter capabilities, and CAS read-back receipt are sealed;
stale, alternate-owner, stopped, or revoked fences are rejected. The authority
consumes an exact paged watcher receipt and a durable checkpoint, then emits
idempotent submission intents only. It has no HTTP client, cancellation path,
or Scheduler write method. A terminal physical
parent recovers only its incomplete logical children with the same seed,
payload, and logical dedupe; completed children are replaced by fresh
stage-local seeds. Tails are deterministically regrouped into at most eight
children, preserving exact quotas `300/150/40/10` and an active target of 500.

The next checkpoint and intent outbox must be durably CAS-committed before an
external authorized driver may POST; submit-before-checkpoint is prohibited.
Immediately before every POST, that driver must re-read the current control
fence and require the same active epoch. The original reconciliation plan,
intent, physical attempt, and exact next-checkpoint outbox remain one harvest
lineage. A POST-before-observation crash therefore replays the committed intent
rather than allocating another seed.

A receipt written before the mutable status cursor is not exposed casually.
After a terminal Scheduler state, a separate GET-only recovery attestation must
scan every ordinal twice/stably, bind present and missing paths, prove one
immutable contiguous receipt prefix, and match the watcher pagination revision.
The harvester re-reads and compares the actual remote SHA/size/mode/value before
publishing the recovered receipt. Runtime parents use a new physical-attempt
dedupe, while the runner and mixed harvester reconstruct and validate the
byte-identical canonical science envelope. Any payload, command, CPU, memory,
child identity, outbox, watcher, or dedupe drift fails closed. Explicit stop is
a monotonic `stopped`/`revoked` CAS fence; an old active checkpoint cannot
produce or submit more refill intents.

## Phase-A PID 55304 authority handoff

Phase B must not race the imported Phase-A driver. The migration state adapter
has no Scheduler mutation endpoint and always emits an empty submission-intent
list.

1. While Phase A is in `batch4` (or any pre-refill phase), Phase B can only seal
   `awaiting_phase_a_refill`; it has no adapter, placement, or authority.
2. After Phase A reaches authenticated `refill`, rerender from the latest driver
   state and allocation snapshot. Import every active Phase-A parent as
   observe-only: it claims one current concurrency slot plus an internal future
   child backlog. Cancellation/replacement before natural terminal is false.
3. Deploy a separate Phase-B-aware consumer in shadow mode. It must authenticate
   both parent protocols, Phase-B manifest/status/receipt files, the mixed
   observe-only adapter, physical parent task IDs, and GET-only Scheduler API
   use. Inventory must use a stable snapshot fence plus descending `before_id`
   pagination and cover the complete ledger. A single `limit=10000` GET is
   prohibited because the live ledger already exceeds 13k rows; capability
   release requires a multi-page 13,000+ row shadow attestation bound to the
   reviewed full-inventory helper commit
   `dafb4509d52f4abcacb68c7bacfc850468d7dc1f`. Its sealed high-watermark,
   `before_id`, total, page count, descending task-ID inventory digest, and
   server snapshot revision receipt are part of the consumer capability. The
   Phase-B adapter executes that exact helper through its read-only
   `list_namespace_tasks` surface, checks all returned IDs are unique and
   strictly descending, rebinds their digest to the receipt, and verifies the
   client's Scheduler POST counter remains zero. The focused shadow fixture is
   13,005 rows over two data pages plus the high-watermark anchor; the same
   attestation remains mandatory against the live ledger before activation.
   The
   existing Phase-A consumer does not
   satisfy this capability by implication.
4. Only after shadow catch-up, write the sealed Phase-A supervisor stop latch.
   Wait for the exact PID 55304 process to exit at its loop boundary.
5. Read the final driver state twice and require identical SHA/revision. Seal an
   exit receipt bound to the stop receipt, PID, and final driver SHA.
6. Re-read allocation capacity and require the exact placement snapshot SHA.
7. Release Phase-B controller authority only when the exit receipt, explicitly
   activation-authorized Phase-B consumer capability, and allocation identity
   all validate. This pure transition still submits zero tasks.
   The sealed handoff records exactly one authority owner: PID 55304 before
   exit, then one Phase-B successor lease afterward. Simultaneous owners are
   prohibited and the released lease still has Scheduler mutation disabled
   until the separate activation authorization.
8. A separately authorized activation may use Phase B only for natural
   deficits. It must not cancel imported Phase-A parents. Stage redistribution
   toward 300/150/40/10 occurs only as old parents naturally terminate.

If Phase A advances after the shadow migration snapshot, refresh/rerender before
writing the stop latch. A stale source driver SHA cannot release authority.

## Rollout and telemetry gates

1. **Offline gate**: focused Phase-B tests, Phase-A compatibility, mixed
   harvester/monitor tests, formatting/lint, deterministic evidence seals, and
   bundle code inventory all pass.
2. **4-child canary**: one 16-CPU/114,688-MiB parent. Require four authenticated
   terminal receipts, disjoint CPU masks, SemLock stress hashes, isolated and
   cleaned child scratch, no shared deletion, and no FEA/AEDT/Scheduler side
   mutation.
3. **Cross-stage 4-child canary**: one parent per stage. Compare per-child wall
   time and parent aggregate child CPU time against the current four-CPU
   reference; require no systematic regression or model-load burst.
4. **8-child canary**: one 32-CPU/229,376-MiB parent per stage. Require exact
   memory/CPU accounting, disjoint masks, staggered loads, no ThreadPool/SemLock
   churn, and sibling survival under one injected seed-local failure.
5. **Tail canary**: exercise representative 44/48/60 CPU packing (8+3, 8+4,
   8+7) and every exact 1..8 resource envelope used by the placement.
6. **Capacity-aware shadow**: rerender the live inventory, prove exact 431+69,
   exact 300/150/40/10 quotas, and zero stage/bundle/wave crossing.
7. **Authority gate**: complete the PID 55304 stop/exit/stable-read/consumer
   capability sequence above, then separately authorize activation.

Every terminal parent records parent wall time, aggregate reaped-child CPU time
(`getrusage(RUSAGE_CHILDREN)` on Linux), requested Scheduler-envelope capacity
seconds, and `CPU/(parent wall * requested CPUs)` utilization. It separately
records summed logical-child elapsed time, logical-child capacity seconds
(`sum child elapsed * 4 CPUs`), and child-work utilization. Dispatch-fill time,
the configured five-second stagger, the 35-second eight-lane bound, envelope
idle-capacity seconds, and zero ready-lane reserve are sealed alongside them.
This preserves the current 4-CPU science contract while creating evidence for
controlled future 1/2/4-CPU canaries. The implementation does not automatically
lower a child to 2 or 1 CPU.

Each child receipt additionally records its seed-runner plus optimizer-process
tree CPU time, child elapsed time, `elapsed * 4` capacity seconds, and the
resulting task-specific utilization. The parent seals the sum of available
child reports and its delta from `RUSAGE_CHILDREN`; this makes low-utilization
tasks directly identifiable instead of inferring them from node averages.

### Finite CPU and density benchmark (design only)

The observed 4-CPU tasks have 37.4% mean busy time across unique nodes, while
1,838 Scheduler CPUs are allocated out of 3,704 physical CPUs. Those are useful
signals but do not prove that allocating more CPUs reduces wall time. Production
therefore remains exactly 4 CPUs per logical child.

The sealed benchmark design uses one held node, identical bundle/model bytes,
seed-local cold caches, and balanced run order:

- CPU scaling: the same three seeds at 4, 6, and 8 CPUs, two paired replicates
  each (18 finite runs).
- Density: the same ten seeds on a 32-CPU parent, comparing exact `8x4` then a
  two-seed tail with an experimental `10x4 on 32` single wave, two paired
  replicates (40 finite runs).
- Every run uses population 320 and 20 authenticated generations. Total: 58
  logical runs, no production policy mutation.

Report wall time, seeds/hour, CPU-hours/seed, task and parent CPU utilization,
p95 child wall time, peak RSS, failures, SemLock attempts, and bit-exact science
identity. A candidate needs at least 10% throughput improvement, no more than
20% CPU-hours/seed or p95 regression, zero failures/SemLocks, and identical
science output. Passing evidence still requires separate authorization; the
benchmark never changes the 4-CPU contract automatically.

Any identity/seal mismatch, stale allocation snapshot, overlapping affinity,
missing SemLock evidence, shared-path cleanup, lost receipt prefix, CPU/memory
accounting mismatch, stale Phase-A driver state, consumer capability gap, or
unexpected Scheduler mutation is a hard stop.
