# Final1000 multi-seed monitoring capability

This change is a local, read-only integration seam. It does not deploy or
restart the 8010 backend, submit or cancel Scheduler tasks, access a remote
account, or use AEDT/FEA.

## Status semantics

`tier1_final1000_multiseed_status.build_compact_snapshot` keeps the hot lane
inventory bounded to 500 physical Scheduler tasks. It publishes separate
counts:

- `physical_lane_count` and legacy `scheduler_task_count`: physical Scheduler
  parents;
- `logical_seed_count`: ordered child slots owned by those parents;
- `logical_sealed_seed_count` / `logical_unsealed_seed_count`: the parent
  journal cursor split;
- `authenticated_terminal_seed_count`: visible immutable seed records.

If a batch parent is present in the hot inventory, a child record is visible
only when `batch_ordinal < sealed_child_count`. A receipt that exists ahead of
the mutable cursor in the crash window is counted in
`hidden_unsealed_seed_record_count` and is not returned to the frontend.

## 8010 adapter

The backend can import:

```python
from tools.tier1_final1000_multiseed_monitor import adapt_condition_index

page = adapt_condition_index(index_path, offset=0, limit=256)
```

The adapter coherently authenticates the compact index, status, shard
manifest, and requested shards. It returns the existing Current7 status schema
and `terminal_results` field without virtual Scheduler task IDs. Every full
normalized response is strictly smaller than 32 MiB.

The equivalent read-only CLI is:

```text
python -m tools.tier1_final1000_multiseed_monitor adapt \
  --index <compact-root>/index.json --offset 0 --limit 256
```

## Production capability gate

The production driver must not enable a compact index merely because the
backend imports. First seal exact adapter files, the deployed backend files,
one exact compact index/status/manifest snapshot, and test-evidence files:

```text
python -m tools.tier1_final1000_multiseed_monitor seal-capability \
  --index <compact-root>/index.json \
  --code-root <clean-release-root> --code-revision <40-hex-commit> \
  --backend-file regression_260707/monitoring/readers.py=<deployed-file> \
  --backend-revision <40-hex-commit> \
  --test-evidence tests/backend-monitor.txt=<test-evidence-file> \
  --test-revision <40-hex-commit> \
  --output <local-capability-receipt.json>
```

Immediately before cutover, call `require_backend_capability(...)` with the
same exact inputs. It refuses a missing receipt, source/backend/test drift,
relocation, or index advancement. Since the index identity is exact, publishing
a newer pointer intentionally requires a new test probe and capability receipt
for a later cutover.

The receipt and capability contract permanently attest:

- no Scheduler mutation;
- no remote write;
- no AEDT use;
- no FEA submission;
- no virtual Scheduler task IDs.
