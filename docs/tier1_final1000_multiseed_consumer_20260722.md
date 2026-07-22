# Final1000 production multi-seed consumer

`tools.tier1_final1000_multiseed_consumer` is the read-only bridge from the
mixed v1/v2 Scheduler campaign to the four compact condition indexes consumed
by the 8010 monitor adapter. It never submits, cancels, or edits a Scheduler
task and never writes to a remote account. AEDT and FEA are outside this
process.

## Safety model

- The default is a write-free shadow preview. `--apply` is required for any
  local publication, and `--watch` is rejected without `--apply`.
- A running v2 parent exposes only child ordinals below its authenticated
  `sealed_child_count`. A receipt visible ahead of that cursor is not
  published.
- Scheduler envelopes, bundle manifests, READY bindings, journal identity,
  result identity, and `(bundle_id, seed)` deduplication are checked before
  any of the four pointers advance.
- All four condition snapshots are prepared first. An identity, schema, or
  result divergence is fatal and leaves every last-good pointer unchanged.
- Only explicit network/transport failures are retried in watch mode. The
  retry budget is five attempts and the poll interval cannot exceed 60 s.
- Physical Scheduler task IDs are retained. No virtual per-seed task IDs are
  generated.

## Shadow canary

Run the command once without `--apply` first. This performs Scheduler GETs and
remote read-only stable reads but creates no lock, lease, cache, or output:

```text
python -m tools.tier1_final1000_multiseed_consumer run \
  --launch-plan <successor-plan.json> \
  --bindings <successor-bindings.json> \
  --controller-state <controller-v2-state.json> \
  --source-launch-plan <v1-source-plan.json> \
  --source-bindings <v1-source-bindings.json> \
  --runtime <live-runtime>
```

For a continuously refreshed shadow index, add an isolated output root and
explicit local-write authorization:

```text
python -m tools.tier1_final1000_multiseed_consumer run \
  <the same authenticated inputs> \
  --output-root <live-runtime>/multiseed-shadow \
  --publish-mode shadow --apply --watch --poll-seconds 15
```

The 8010 backend can read each resulting
`conditions/<stage>/canonical/current7-index.json` with
`tier1_final1000_multiseed_monitor.adapt_condition_index`.

## Canonical cutover

Canonical mode is intentionally unavailable while the v1 harvester owns the
four live pointers. The 15-second consumer default is half the production
driver's 30-second reconcile interval so a newly exported controller state can
be observed before the next refill reconcile. Cutover requires this sequence:

1. Ask the v1 harvester to stop through its authenticated stop-file contract.
2. Verify the old PID has exited and the four v1 pointers are stable.
3. Preview a handoff receipt with `seal-v1-handoff` (write count remains zero).
4. Repeat with `--apply` to write the exact handoff receipt.
5. Let the production driver persist `cutover_prepared` and its atomic
   `--prepared-controller-state` export. This phase performs refill POST0.
6. Start one canonical consumer against that exact controller export with
   `--handoff-receipt`, `--apply`, `--watch`, and `--poll-seconds 15`. Its
   nonblocking OS lease refuses a concurrent writer.
7. Pass its capability path to the driver with `--consumer-capability`. Only a
   fresh canonical receipt for the current controller SHA releases refill.

Canonical publication also requires `--output-root` to equal `--runtime`.
After each successful cycle, the consumer writes a freshness-bounded
capability receipt that exact-binds the inventory and all four condition
indexes. `check-capability` is the driver gate. The driver repeats this check
before every new reserve/POST: a valid receipt for the prior controller SHA is
treated as normal `capability_catching_up` and performs POST0, while stale,
shadow, tampered, or unlocked receipts fail closed. A clean stop writes a
separate restart handoff receipt; `--previous-stop-handoff` verifies it before
the next writer starts.

No canonical cutover or live process stop is performed by this change.
