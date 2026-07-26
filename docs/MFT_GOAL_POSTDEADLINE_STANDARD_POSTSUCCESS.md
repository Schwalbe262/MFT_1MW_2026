# Post-deadline Standard post-success bridge

`tools.mft_goal_postdeadline_standard_postsuccess` watches the immutable
collector outputs for tasks 96325 and 96327. It does not call Scheduler; the
two source collectors remain the only Scheduler readers and use GET only.

Before accepting a result, the bridge reauthenticates:

- the custom plan, submission receipt, candidate SHA, effective FEA
  parameters, solver/library revisions, and terminal-success task snapshot;
- the collected `RESULT_JSON`, retained symmetric AEDT, transport chunks,
  AEDT-results manifest, and prune-protection marker;
- the exact fixed operating/cooling identity, including dual fan at 1.5 m/s,
  TIM/insulation conductivity 0.2 W/(m K), and both 2 mm cooling pads.

It then evaluates the measured exterior dimensions, minimum self resonance,
all active winding/core body and probe temperatures, and writes a sealed
diagnostic observation. The limits are exactly 1200/1000/750 mm, 15 kHz,
100 C winding, and 120 C core.

Available actual observations receive an exact two-objective non-dominated
sort over measured volume and measured total loss. This diagnostic measured
front is not a production Pareto front. Measured observations are not directly
unioned with surrogate rows from the 512-seed aggregate.

The named adapter is also registered with `mft_goal_strict_al_ingest`. Every
successful row must pass the existing 25-target strict-quality checks and the
pinned base-dataset cohort. Retraining remains closed until the existing
8-row, 8-geometry, 4-source gate passes. No dataset or model is written by the
watcher.

The live watcher started for the two tasks writes beneath:

```text
C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\
postdeadline_standard_postsuccess_96325_96327_v1
```

While neither collector has published a success collection, `state.json`
contains `status=pending_standard_collections`, `pending_count=2`, and all
scientific, production, and production-Pareto claim fields remain `false`.
