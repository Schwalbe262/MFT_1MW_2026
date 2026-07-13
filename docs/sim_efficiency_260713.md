# 1 MW MFT simulation-efficiency A/B plan (2026-07-13)

## Executive finding

The measured optimization order is Icepak, the Maxwell loss solve, and then
everything else. In 1,330 valid rows from the pinned current solver revision at
the production settings, thermal consumed 62.9% and loss 32.9% of recorded
stage time. Matrix consumed only 4.1%. The current `time` field is not process
wall-clock: it excludes EM model construction, EM extraction, project saves,
result persistence, and shutdown.

No efficiency setting is approved from projections alone. Use paired designs,
the harness below, and the accuracy gate before changing campaign defaults.

## Timing evidence

The source was the read-only campaign snapshot under
`Y:/git/MFT_1MW_2026/regression_260707`: 360 collected parquet files, 3,870
snapshot rows, and 1,889 unique result identities after collector-snapshot
deduplication. All files had only `time_matrix`, `time_loss`, `time_thermal`,
and `time`; there was no modeling/startup/save/extraction breakdown.

The strict current-revision cohort (`b171c7c...`) uses matrix 1.5%/20/1,
loss 1.5%/10/2 with copy reuse, thermal cap 250, and Rx skin mesh. Both EM and
thermal validity gates passed for all 1,330 rows below.

| recorded stage | mean min | p25 | p50 | p75 | p90 | aggregate share |
|---|---:|---:|---:|---:|---:|---:|
| Matrix `Analyze` | 4.28 | 2.40 | 3.65 | 5.17 | 7.38 | 4.1% |
| Loss `Analyze` | 34.10 | 17.47 | 27.15 | 43.97 | 67.31 | 32.9% |
| Thermal build + setup + solve + extraction | 65.11 | 37.01 | 55.94 | 85.14 | 114.35 | 62.9% |
| Recorded total | 103.48 | 66.56 | 93.29 | 130.99 | 175.37 | 100% |

Convergence telemetry explains some headroom. Matrix used a median 11 passes
(p90 12), loss used four passes from p25 through p95, and thermal used a median
139 iterations (p90 168, p95 176). Icepak already has native convergence
criteria of flow `1e-3` and energy `1e-7`; 250 is a ceiling, not a requirement
to execute 250 iterations.

The latest API window could join only 13 completed valid tasks to result rows,
so the end-to-end figures are indicative: scheduler `started_at` to
`finished_at` had a 70.55-minute median, and the residual above recorded `time`
had p25/p50/p75 of 4.52/6.04/8.19 minutes. The residual was 9.83% in aggregate
and bundles startup, modeling, extraction, saves, persistence, and shutdown.
Forty-three deduplicated AEDT-start log events had a 10.82-second median and
11.91-second p90, so single-design AEDT startup is not a leading target.

`failed_samples_260706.jsonl` has 59 records but only a failure timestamp, not
elapsed time. It cannot quantify failed-job waste. The new instrumentation will
make future successful A/B rows separable; failure timing would need a separate
failure-record extension.

## Added timing instrumentation

`run_simulation_260706.py` now emits additive `mft-stage-timing-v1` columns.
Existing `time_matrix`, `time_loss`, `time_thermal`, and `time` retain their old
meaning and therefore do not break existing collectors.

The new fields cover:

- Python/import-to-run delay, AEDT Desktop startup, and project/input setup;
- matrix model construction, solver dispatch, post-analyze overhead, and result
  extraction;
- copied-loss preparation (or independent loss model construction), solver
  dispatch, native pre/post overhead, and extraction;
- cumulative project-save call count and elapsed time before result assembly;
- total thermal time plus `thermal_build_s`, `thermal_setup_s`,
  `thermal_solve_s`, and `thermal_extraction_s`;
- run-to-result, process-to-result, unattributed time, and UTC boundary
  timestamps.

The A/B launcher adds `ab_process_wall_s`, measured around the entire solver
subprocess. That is the preferred timing comparator because it includes result
persistence and cleanup. The stage fields diagnose why it moved.

## Ranked speedup candidates

Savings are projections against the measured median stage times, not solved
claims. The ranking favors expected time saved per accuracy risk.

| rank | variant to test | projected saving | accuracy/robustness risk | gate and rationale |
|---:|---|---:|---|---|
| 1 | Multi-turn Rx side-block Icepak mesh level 5 to 4 (`thermal_rx_side_block_mesh_level=4`) | 10-30% of thermal on side-pack designs, about 5.6-16.8 min at the cohort median | Low-medium | Existing branch `1071b3c` documents level 4 as the restored-zone floor for 1.621 mm homogenized packs. Exact singleton 0.300-0.435 mm turns remain forced to level 5. Require every thermal validity/power-balance/residual gate and every Tprobe within 2 C. |
| 2 | Loss `percent_error=2.0`, `max_passes=8`, keep `min_converged=2` | 15-35% of loss, about 4.1-9.5 min at median | Medium | Loss currently settles at four passes for at least p95 of valid rows, but mesh adaptation can move local copper/proximity loss. Require all `P_*` loss outputs within 2%, B within 2%, and temperature within 2 C. |
| 3 | Loss `rx_mesh_mode="length"` (do not start with `length-coarse`) | 10-25% of loss, about 2.7-6.8 min at median | Medium | Removes skin-depth refinement on Rx foils and can bias proximity loss. `length-coarse` projects 15-40% but is medium-high risk; test it only after `length` passes. |
| 4 | Remove only saves proved redundant by timing, starting with the ordinary matrix post-analyze save before the strict pre-copy save and the disposable final fixed-mode save | 1-5% of whole pipeline, roughly 1-5 min; highly file-system dependent | Low-medium | A representative replay logged nine project saves. Preserve strict saves around Copy/Paste validation and native loss dispatch. Implement behind an A/B control and verify copied-design integrity plus result persistence before adoption. |
| 5 | Matrix `matrix_percent_error=2.0`, `matrix_max_passes=16`, keep `matrix_min_converged=1` | 15-35% of matrix: about 0.5-1.3 min at median, 0.4-2.6 min at observed p25-p90 | Low-medium | Matrix is only 4.1% of recorded time. Do not use an eight-pass cap: an earlier random case stopped with 13.254% energy error. Require Llt and k within 0.5% and convergence telemetry valid. |

Lowering `thermal_max_iterations` to 200 is a failure-tail/yield experiment,
not an accuracy-free early-exit. It saves nothing when Icepak converges before
200, while truncating the slow cases. Current valid p95 is 176 iterations, so a
200 cap can be tested after mesh A/Bs, with convergence failures counted as A/B
failures rather than silently discarded.

`--headless` already maps to `non_graphical=True` and is present in campaign
commands. Reusing one AEDT Desktop across designs could save roughly 11 seconds
of measured startup per design, but adds cross-design state/crash risk and is
not competitive with thermal/loss work.

## A/B harness

Use one unchanged design JSON and a small overlay. A plain overlay modifies the
variant only; the structured form `{"baseline": {...}, "variant": {...}}`
can pin controls in both arms. The launcher does not contact the scheduler.

Example mesh overlay:

```json
{
  "thermal_rx_side_block_mesh_level": 4
}
```

Run both arms sequentially in one allocation:

```bash
python regression_260707/verify/run_efficiency_ab.py \
  --params cand.json \
  --overlay mesh_l4.json \
  --arm both \
  --output-dir ab/mesh_l4/design_001
```

For separate cluster tasks, submit the same command and isolated output path
once with `--arm baseline` and once with `--arm variant`. This avoids one arm
consuming the other's task timeout. The output directory contains effective
parameter JSONs, streamed logs, result JSONs, and a manifest with parameter
SHA-256 identities. `ab_process_wall_s`, arm identity, and a common experiment
SHA-256 are embedded in each result, which is also re-emitted as an unprefixed
`RESULT_JSON` line for the standard scheduler collector. A failed rerun removes
any stale result before launch.

Compare the two results:

```bash
python regression_260707/verify/compare_efficiency_ab.py \
  ab/baseline_result.json ab/variant_result.json \
  --json-output ab/comparison.json
```

The comparator also accepts CSV and selects the last row by default; use
`--baseline-row`/`--variant-row` to choose another row. It gates Llt/k at 0.5%,
all finite `P_*` loss targets at 2%, all finite B targets at 2%, and Tprobe
targets at 2 C. `P_target` is excluded. Missing required families fail closed;
optional targets that are non-finite in both arms are shown as skipped. Override
limits with `--em-relative-pct`, `--loss-relative-pct`, `--b-relative-pct`, and
`--temperature-absolute-c`. Solver validity, thermal convergence/extraction/
power balance, subprocess return code, matching experiment identity, and
baseline/variant arm order all fail closed.

Use several geometrically diverse designs, keep node/resources/revisions
identical, and alternate arm order when both run in one allocation. Report
accuracy failures and convergence/yield failures as well as medians; comparing
only successful fast variants would bias the decision.

Use `--order variant-first` for the reversed sequential order.

## Scheduler and queue bottlenecks

The latest 200 `mft*` tasks contained 134 successful campaign tasks suitable
for lifecycle statistics. Created-to-attached wait had a 30.2-minute median and
54.1-minute p90. This is predominantly allocation/capacity latency, not node FEA
wall-clock. Once attached, attached-to-started took a 5-second median and
7-second p90.

Three consecutive completed health ticks took 38.0, 42.6, and 69.2 seconds.
`refresh_allocations` plus `maintain_allocation_pool` consumed 62.3% of their
aggregate duration. A later single tick took 152.0 seconds and was instead
dominated by `orphan_process_sweep` (68.9 s), `refresh_cluster_state` (30.5 s),
and pool maintenance (28.7 s). The prompt's earlier approximately 8.4-second
allocation refresh and 6.2-second task refresh plus these samples show high
variability. `/api/health` exposes only the most recent tick, so this is a
ranking signal rather than a stable benchmark.

Concrete scheduler-repository suggestions (not implemented here):

1. Fetch one batched Slurm/account state snapshot per tick and share it among
   cluster-state, allocation-refresh, pool, and task stages. Update only changed
   database rows.
2. Refresh only active/stale allocations; batch remote polling or use bounded
   parallel I/O where one query cannot cover the set.
3. Run pool maintenance on a deficit/state-change trigger or slower cadence,
   with capacity pre-indexed by account/capability/node.
4. Bulk-query tasks/jobs by account, batch writes, and index active-status,
   allocation, and updated-at lookups. Build ready-task/free-capacity indexes
   once per tick instead of rescanning them for each assignment class.
5. Instrument orphan sweeping by host and process count, bound each remote
   probe, and move slow cleanup off the assignment-critical path.
6. Split fast assignment ticks from slower reconciliation/pool maintenance so a
   38-152 second maintenance tick cannot delay already-ready work.
7. Publish rolling stage p50/p90/count plus remote-call and rows-scanned metrics.
   Persist `ready_at`, first-fit, assigned-at, and queue-reason transitions so
   capacity wait can be separated from avoidable scheduler latency.

The latest-200 window is truncated and includes running tasks, terminal queue
reasons are blank, and task timestamps have one-second resolution. Queue figures
must therefore remain separate from per-design node wall-clock.
