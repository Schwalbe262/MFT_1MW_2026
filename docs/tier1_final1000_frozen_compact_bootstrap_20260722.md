# Final1000 frozen compact-v2 bootstrap

This bootstrap removes the ordering dependency between the 8010 backend and
the future multi-seed consumer/controller. It creates a deterministic compact
v2 index locally before either writer exists. It does not integrate or restart
the current `fc51` backend and performs no Scheduler, remote, AEDT, or FEA
operation.

## Probe contract

The frozen fixture contains two positive physical Scheduler-style task IDs:

- one completed `final1000-single-seed-v1` lane;
- one running batch-4 lane with two sealed children, one deliberately
  unsealed receipt, and one child slot not yet produced.

The normalized page therefore proves `physical=2`, `logical=5`,
`sealed=3`, `unsealed=2`, `hidden_unsealed_record=1`, and three visible
terminal records. No virtual task ID is created. The compact status, index,
shards, normalized page, JSON evidence, and JUnit evidence are all checked
against the exclusive 32-MiB bound.

All immutable files have content-addressed names. The bootstrap manifest also
records its absolute publication root. Copying it to another location is an
intentional capability failure; create it directly in the final local runtime
root instead.

## Exact future fc51-successor commands

Run these commands from the clean candidate release containing this module.
`$BackendRevision` must identify the exact reviewed successor backend tree,
not the current `fc51` deployment.

```powershell
$ReleaseRoot = 'C:\path\to\clean\candidate-release'
$ProbeRoot = 'C:\path\to\future-8010-runtime\compact-v2-bootstrap'
$CapabilityRoot = 'C:\path\to\future-8010-runtime\capabilities'
$CodeRevision = (git -C $ReleaseRoot rev-parse HEAD).Trim()
$BackendRevision = '<40-or-64-hex-reviewed-successor-revision>'
$TestRevision = $CodeRevision

$CreatedJson = python -m tools.tier1_final1000_multiseed_monitor_bootstrap create `
  --output-root $ProbeRoot
$Created = $CreatedJson | ConvertFrom-Json
```

Pass every backend file imported by the future compact-v2 route. The logical
name before `=` is the exact deploy-relative name that is sealed into the
receipt; the path after `=` is the exact candidate file whose bytes are
tested. Add another `--backend-file` for each file in the backend closure.

```powershell
$SealedJson = python -m tools.tier1_final1000_multiseed_monitor_bootstrap seal-capability `
  --output-root $CapabilityRoot `
  --bootstrap-manifest $Created.bootstrap_manifest_path `
  --code-root $ReleaseRoot `
  --code-revision $CodeRevision `
  --backend-file regression_260707/monitoring/readers.py=C:\path\to\successor\regression_260707\monitoring\readers.py `
  --backend-file regression_260707/monitoring/app.py=C:\path\to\successor\regression_260707\monitoring\app.py `
  --backend-revision $BackendRevision `
  --test-revision $TestRevision
$Sealed = $SealedJson | ConvertFrom-Json
```

The seal automatically binds the exact default adapter/runtime closure
reported by `default_adapter_code_files`, the caller-supplied backend file set
and revision, both generated test-evidence files (`.json` and `.junit.xml`),
and the exact frozen index/status/manifest identity.

Immediately before an independently reviewed deployment, validate the same
inputs again:

```powershell
python -m tools.tier1_final1000_multiseed_monitor_bootstrap validate-capability `
  --receipt $Sealed.receipt_path `
  --bootstrap-manifest $Created.bootstrap_manifest_path `
  --code-root $ReleaseRoot `
  --code-revision $CodeRevision `
  --backend-file regression_260707/monitoring/readers.py=C:\path\to\successor\regression_260707\monitoring\readers.py `
  --backend-file regression_260707/monitoring/app.py=C:\path\to\successor\regression_260707\monitoring\app.py `
  --backend-revision $BackendRevision `
  --test-revision $TestRevision
```

Validation fails closed when adapter/backend/evidence bytes change, the
publication moves, or a different coherent index is supplied. A later compact
index intentionally needs a new probe and capability seal.

## Scope boundary

The bootstrap only writes beneath explicit `--output-root` values. It neither
claims that the current `fc51` backend consumes compact v2 nor authorizes a
deployment/restart. Backend integration, isolated import tests, and a separate
release review remain required before changing port 8010.
