# Final1000 task 87052 paired resource comparator

## Current disposition

**Launch is forbidden. Do not submit either companion from this package.**

The package was prepared offline for the exact Phase-B entry seed
`2257498123`, but the intended 4-CPU anchor, Scheduler task `87052`, reached
terminal `failed` at `2026-07-22 23:50:37` with exit code 1:

```text
RuntimeError: Phase B parent memory evidence seal mismatch
```

That GET-only observation is sealed into both the config and package. The
submission function checks `launch_policy.launch_allowed` before any Scheduler
GET or POST and therefore cannot launch this revision.

The earlier unpaired 2-CPU task 86501 is also bound as negative rationale. It
completed science but was not promotable: 4,600.787 s versus the unrelated
4-CPU baseline's 1,871.530 s, single-seed throughput ratio 0.4068,
slot-weighted ratio 0.6325, average 1.2403 cores, core-use ratio 0.8183, and
peak RSS 18,983,600,128 bytes. Because that comparison used another seed, it
cannot distinguish seed complexity from CPU scaling; it must not decide the
production CPU shape.

## Sealed offline artifacts

- Config:
  `docs/evidence/tier1_final1000_paired_resource_comparator_87052_20260723.json`
  - canonical config SHA: `dca22b9c40b833dbc0e9600b736bf0542cb7bcbaaa0d79b257b9c5cb880d3ecb`
  - file SHA: `49eeebd13e46708adc382f8d726dd05eca2b022e781d5c3af3878bf23888f5d6`
- Package:
  `docs/evidence/tier1_final1000_paired_resource_comparator_87052_package_20260723.json`
  - canonical package SHA: `50a71468bc1abed323f4bcfe479aa29a6deb8a37a8e668a95a242a1e5019d1ac`
  - comparator identity SHA: `4ca2f2304007a929c3f202cab89bca9067f3263aa53c8c153b358423bb6e6cfa`
  - file SHA: `fbc8b9d07f5b3963a9637928183c9365f770505d98ccefc57c91633b5f9c4ce4`
- Tool: `tools/tier1_final1000_paired_resource_comparator.py`
- Tests: `tests/test_tier1_final1000_paired_resource_comparator.py`

The package authenticates the exact task-87052 source bundle/package:

- bundle `current7-c2d90bf33d513c607f8f`
- stage `entry-1200-t125`
- seed `2257498123`
- source parent task SHA
  `69263c2ad94dfd32d212c25c7072c7b9c3a477facc753310f75271a41fb123bf`
- source child task SHA
  `95f458f11a7cd2063908e304bc3fe04a182d5dff8fdfb2fe6a9aac4e9f66794c`
- source child payload SHA
  `6fb21e402dfac0e946748a0d3850c510a7e1d13f64733b0b3b5924bf9019003f`

The prepared 2- and 3-CPU payloads are byte-for-byte identical to the source
child science payload except for `/inference_threads` and
`/scheduler_cpus`. At the Scheduler envelope, only `name`, `dedupe_key`,
`command`, `cpus`, and those two nested resource fields differ. Memory remains
28,672 MiB, optimizer processes remain one, max workers remain 32, and the
bundle/stage/seed/models/data/hard constraints/warm start all remain exact.

## Capacity and decision rule

The read-only allocation snapshot at `2026-07-23T08:37:11+09:00` contained 41
active, non-exclusive CPU allocations. Each allocation is packed independently
with:

```text
slots(c) = min(floor(total_cpus/c), floor(total_memory_mb/28672), 32)
```

This gives exact same-shape capacities:

| CPU per seed | Exact slots |
| ---: | ---: |
| 2 | 719 |
| 3 | 567 |
| 4 | 459 |

After a valid replacement anchor and both companions complete, the only
performance decision metric is:

```text
allocation-local exact slots / measured per-seed wall seconds
```

A 2- or 3-CPU shape can be recommended only when all resource gates pass, all
three normalized science fingerprints are identical, and its slot-weighted
throughput is strictly greater than the 4-CPU anchor. Per-seed wall time is
reported, but there is no independent per-seed latency gate and no latency
override. A tie retains 4 CPU. The evaluator never promotes production
automatically.

If a later independent review promotes a shape, the batch target is the full
current allocation-local exact fit for that shape—not 500 and not a fixed
single-task count. The controller must recompute active CPU allocations each
cycle, respect each account/allocation's CPU and memory boundary plus the
32-worker node cap, and continuously refill to
`exact_fit_slots[selected_cpus]` until stop. It may not pack aggregate CPU or
memory across allocation boundaries. Admission remains one durable reservation
and one POST at a time so a failed reservation cannot leak a burst of
additional submissions. Before paired evidence exists, 4 CPU is the fallback;
the failed unpaired 2-CPU result explicitly forbids a 2-CPU rollout.

## Telemetry collected by each companion

The command embeds a self-contained wrapper because the immutable pbd8 bundle
does not contain this new tool. It records and seals:

- exact inherited and final `sched_getaffinity` CPU sets;
- exact `SLURM_CPUS_PER_TASK` plus OMP/OpenBLAS/MKL/NumExpr/VecLib bindings;
- bounded raw `/proc/self/cgroup` and `/proc/self/mountinfo` evidence;
- validated cgroup v1/v2 membership and nearest finite memory ancestor;
- cgroup current/peak/limit with request coverage and limit gates;
- process-tree CPU seconds, wall seconds, average cores, utilization;
- independent process peak RSS capped at 22 GiB;
- seed status and result file SHA/size;
- a normalized science fingerprint which removes only resource-binding fields,
  while retaining the optimizer/Pareto artifact fingerprints.

It invokes the existing single-seed runner directly. It does not invoke the
hard-coded 4-CPU Phase-B parent, and has no FEA or AEDT path.

## Requirements before any future launch

All of the following require evidence and a new package revision; an operator
must not edit this package to flip `launch_allowed`:

1. Fix the Phase-B parent-memory evidence contract and independently test the
   exact failure that terminated task 87052.
2. Run a new, isolated 4-CPU anchor using the same pbd8 bundle, stage, seed,
   optimizer process count, 28,672 MiB request, and 4-thread resource binding.
3. The replacement anchor must finish `completed`, produce a validated result,
   expose exact 4-CPU affinity/process-tree CPU/RSS/cgroup evidence, and pass
   all terminal gates. A failed task or a result recovered from a failed parent
   is not an anchor.
4. Regenerate the comparator config/package with the replacement Scheduler task
   id, name, dedupe, source receipt, terminal SHA, and complete namespace
   snapshot. The present package identity must be superseded, not reused.
5. Obtain an independent, package-SHA-bound approval for each companion
   separately. Each approval permits exactly one CPU shape and one maximum
   Scheduler POST. No automatic retry is allowed; a retry needs another new
   package and approval.
6. Immediately before each POST, re-authenticate live READY, the exact completed
   replacement anchor detail, and complete paged `mft-t1fg-` plus comparator
   namespaces. The selected name/dedupe must be absent and every same-seed row
   must be an authenticated member of this comparator.
7. After each POST, re-scan the complete namespaces and prove exactly one new
   task id and one exact candidate identity. Never cancel, preempt, reserve an
   allocation, publish a bundle, submit FEA, or start AEDT from this workflow.

An independent approval must be a canonical JSON object using schema
`mft-tier1-final1000-paired-resource-comparator-approval-v1`, bound to one
future package SHA, one completed anchor remote-terminal SHA, and exactly one
of `[2]` or `[3]`. It must attest source/companion identity review, complete
namespace policy review, accepted terminal assumptions, maximum one POST,
new-package-on-retry, and `automatic_promotion_allowed: false`.

## Safe offline checks

Validate the current blocked package (no network or mutation):

```powershell
python tools/tier1_final1000_paired_resource_comparator.py validate `
  --package docs/evidence/tier1_final1000_paired_resource_comparator_87052_package_20260723.json

python -m pytest -q tests/test_tier1_final1000_paired_resource_comparator.py
```

Do not run `submit --apply` with this revision. It is expected to fail closed
before Scheduler access because task 87052 is a failed anchor.
