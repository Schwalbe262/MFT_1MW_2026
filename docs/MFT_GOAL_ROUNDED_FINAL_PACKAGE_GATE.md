# Rounded candidate-5 final package gate

`tools/mft_goal_rounded_final_package_gate.py` is the fail-closed package
boundary for the final R10/s4 design. It is a new schema and code path. It
does not import, modify, downgrade, or publish through the legacy
task96324/task96326 `mft_goal_final_package_gate.py`.

The tool currently exposes only `prepare` and `validate`. Neither command
creates a delivery package, copies payload files, calls Scheduler, or
publishes anything.

## Authorities and classifications

The candidate identity is pinned to physics SHA-256
`909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42`.
The model contract is:

- `round_corner=1`;
- corner radius 10 mm;
- four chord segments per quarter corner;
- dual fans at 1.5 m/s;
- TIM/insulation conductivity 0.2 W/(m K);
- 2 mm core-plate pads; and
- 2 mm WCP pads.

Only the authenticated terminal-success task96340 Standard collection can
set `symmetric_scientific_pass=true`. A diagnostic symmetric AEDT can be
inventoried as a file-only fallback, but it always sets
`scientific_package_allowed=false`.

A terminal rounded Full collection completes the symmetric-to-Full
cross-check. The gate independently checks its collection seal, terminal
task, result, fixed boundary, rounded identity, actual dimensions,
resonance, winding/core temperatures, AEDT, and native results manifest. A
pre-solve Full geometry/setup checkpoint can still occupy
`models/full_model.aedt` for model-file delivery, but it is permanently
classified:

- `scientific_result_available=false`;
- `scientific_pass=false`;
- `full_crosscheck_complete=false`; and
- `diagnostic_only=true`.

The global search authority is the exact
`aggregate_rolling512_d4e4d60` result: 512 distinct seeds, 163,840 terminal
rows, 133,563 deduplicated geometries, exact all-row global NDS, zero
production-feasible Pareto points, 22 audit-objective-front points, and 12
deterministic Standard candidates. The gate hashes the 1.45 GB
`global_terminal_candidates.csv` as well as the production Pareto CSV, audit
front CSV, Standard candidate CSV, aggregate manifest, and interactive
Pareto HTML. It rejects seed-local Pareto merging.

## Prepared package inventory

The planned package contains SHA-256 and byte-size records for:

- `models/symmetric.aedt`;
- `models/full_model.aedt`;
- symmetric and Full result JSON, collection seals, Scheduler provenance,
  and native AEDT results-tree manifests when scientifically available;
- candidate-5 rounded parameters, search-selection evidence, execution
  profile, and rounded drawing dimensions;
- global NDS/Pareto CSVs, aggregate manifest, and Pareto HTML; and
- final rounded drawing PPTX/PDF when supplied.

The prepare plan also pins future package schemas
`mft-goal-rounded-final-delivery-package-v1` and
`mft-goal-rounded-final-delivery-package-seal-v1`. Their atomic contract
requires a same-filesystem staging directory, source-byte reauthentication,
a complete manifest inventory, a seal over that inventory, and one final
`os.replace`. Partial destinations are forbidden. No publisher is included
in the current tool.

## Terminal Full usage

After task96340 and the rounded Full collector both finish:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_rounded_final_package_gate.py prepare `
  --standard-plan <rounded-final-prepare-plan.json> `
  --standard-final <rounded-submission-final.json> `
  --standard-collection <task96340-authenticated-collection> `
  --full-terminal-collection <rounded-full-terminal-collection> `
  --aggregate-root <aggregate_rolling512_d4e4d60> `
  --pareto-html <global-pareto-audit.html> `
  --drawing-pptx <final-rounded-drawing.pptx> `
  --drawing-pdf <final-rounded-drawing.pdf> `
  --output <rounded_final_package_prepare.json>
```

Reauthenticate the sealed plan and all source bytes without publishing:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_rounded_final_package_gate.py validate `
  --plan <rounded_final_package_prepare.json> `
  --report <rounded_final_package_validation.json>
```

## Full checkpoint usage

If only the diagnostic Full checkpoint is available, replace
`--full-terminal-collection` with:

```text
--full-checkpoint-collection <diagnostic-checkpoint-collection>
```

The command may prepare a model-delivery inventory, but its readiness records
that the Full cross-check is incomplete and that a scientific final package
is forbidden.

If the symmetric scientific collection is unavailable, omit both
`--standard-final` and `--standard-collection`, then provide:

```text
--symmetric-fallback-aedt <diagnostic-symmetric.aedt>
```

That fallback is also model-only and can never satisfy the scientific
package gate.
