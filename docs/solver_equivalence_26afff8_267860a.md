# Solver physics-equivalence review: 26afff8 to 267860a

The solver revision `267860a86dc8c8017c4b713f6674c0614cc365ce` is
approved as data-physics equivalent to
`26afff8de2936f605783395fbff19d5f1d26b354` for the existing
`mft1mw-1k101-native-lamination-kf0p85-v3` cohort.

The reviewed diff changes pooled-AEDT lifecycle behavior only:

- serializes Desktop-global attach/model/save/extraction operations;
- yields that lock only for an exact project/design native solve;
- rebinds exact native project/design handles after sibling activity;
- preclaims and safely reclaims empty cross-account `.aedtresults`
  directories; and
- adds controller/monitoring evidence for those lifecycle changes.

No geometry builder, input range, material value, Maxwell/Icepak setup
parameter, excitation expression, loss expression, or
`PHYSICS_DATA_REVISION` is changed.  New branches in the solve and thermal
paths are guarded by the pooled backend; the standalone physics path remains
unchanged.  The raw native `Analyze` calls use the same already-created setup
and DSO profile.

Review commands:

```powershell
git diff --stat 26afff8de2936f605783395fbff19d5f1d26b354..267860a86dc8c8017c4b713f6674c0614cc365ce
git diff 26afff8de2936f605783395fbff19d5f1d26b354..267860a86dc8c8017c4b713f6674c0614cc365ce -- run_simulation_260706.py module/thermal_260706.py module/aedt_pool_adapter.py
python -m unittest regression_260707.test_pipeline_completion.PipelineCompletionTests.test_pooled_release_reuses_reviewed_same_physics_rows
```

The equivalence is deliberately directional: collection pinned to `26afff8`
may reuse clean `267860a` rows with the exact physics-data revision and pinned
library.  Dirty worktrees, a foreign physics revision, or unrelated solver
SHAs remain quarantined.
