# B171 mature refill controller release

This branch preserves the revision-pinned B171 continuous refill controller
from a clean GitHub clone. It is a source release only: preparing, auditing,
testing, committing, or checking out this branch does not execute scheduler
queries or mutations.

## Mature production health policy

The strict collector remains the training-data quality boundary. Once the
campaign is promoted to mature production, the controller keeps the requested
logical pool full while reporting these conditions under
`production_nonblocking_alerts`:

- individual failed, timed-out, nonconverged, or strict-invalid samples;
- recent/fleet valid-rate degradation;
- revision mismatch and thermal saturation findings;
- strict collector pin/refresh lag;
- repeated informative runtime fingerprints.

These findings do not populate `pause_reasons`, and therefore do not stop
replacement submissions. Malformed evidence, identity/seal drift, stale
durable-state transitions, invalid project-cap contracts, and scheduler API
exceptions are not caught by this policy and still fail closed.

## Reproducible evidence

Static audit reads only repository-owned source and immutable artifacts:

- the exact production, predecessor, recovery, and rollback manifests;
- a sealed reviewed local-recovery evidence record containing the original log
  SHA-256 and the exact fields reviewed from its single `RESULT_JSON`;
- a sealed reviewed record of the never-started rejected submission.

The historical mounted paths in those evidence records are provenance strings,
not runtime fallbacks. Static audit never opens them. Historical scheduler-fix
file hashes are covered by their embedded incident audit seal; current mutable
scheduler source is not re-read.

## Verification

From the repository root with the production Python environment:

```powershell
python -m py_compile regression_260707/campaign/_continuous_refill_b171c7c.py regression_260707/campaign/_adopted_refill_sha688c6f9.py regression_260707/campaign/_rolling_recycle_prebinding_260712.py regression_260707/campaign/_submit_production300_b171c7c.py regression_260707/campaign/feeder.py regression_260707/campaign/rapid_campaign.py regression_260707/verify/scheduler_client.py
python regression_260707/campaign/tests/test_adopted_refill.py
python regression_260707/campaign/tests/test_continuous_refill_incidents.py
python regression_260707/campaign/tests/test_mature_refill_policy.py
python regression_260707/campaign/_continuous_refill_b171c7c.py
```

The three unittest files contain 31 tests. The final command is read-only and
must report `mode: static_audit_only`,
`scheduler_query_count: 0`, and `scheduler_mutation_count: 0`.

Activation remains a separate reviewed operation and requires both the exact
plan SHA and the explicit dynamic project-cap authorization flag.
