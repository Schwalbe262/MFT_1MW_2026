# MFT thermal strict-yield evidence, 2026-07-13

## Outcome

The supplied `strict_data_status.json` snapshot contains 1,014 strict-full rows
out of 1,543 raw rows (65.7%). That aggregate mixes obsolete revisions and
analysis profiles. When the currently pinned production revision is evaluated
as the trusted revision, its observed strict-full yield is already above the
restart target: 1,330 / 1,424 rows (93.4%). The restart pilot still has to prove
that this rate carries forward to the new solver revision.

This work hardens genuine probe-extraction weak points without relaxing the
quality contract. It does not relabel missing thermal fields as valid and does
not substitute a surface probe for a missing modeled-volume maximum.

## Evidence and root causes

Read-only sources inspected:

- `Y:/git/MFT_1MW_2026/regression_260707/training/strict_data_status.json`
- `Y:/git/MFT_1MW_2026/regression_260707/data/dataset/train.parquet`
- campaign logs and `continuous_refill_b171c7c_state.json` under the supplied
  regression tree
- the quality-contract, collector, scheduler, input, modeling, and thermal
  extraction paths in the campaign and this worktree

No `failed_samples*.jsonl` file was present in the supplied regression tree.

### Thermal nonfinite, convergence, and extraction categories

The snapshot's large thermal reason counts overlap. In particular, a rejected
solve is intentionally not queried: `run_thermal_analysis` returns NaNs after
convergence validation fails. Therefore a Tprobe nonfinite reason does not, by
itself, prove a probe-query failure.

For the pinned `b171c7ce5f7a018be6a575a32b1a1f5b7caa980c` cohort:

| Converged | Extraction complete | Rows | Interpretation |
| --- | --- | ---: | --- |
| yes | yes | 1,330 | strict-full valid |
| no | no | 85 | extraction correctly skipped |
| yes | no | 9 | genuine post-solve field omission |

The 85 solve-gated rows comprise 77 `residual_threshold` cases at the 250
iteration ceiling, six `monitor_missing` cases after failed dispatch, and two
`monitor_malformed` cases. These are convergence/solver-evidence failures, not
probe extraction failures.

All nine converged extraction failures have `N2_side=1`. Every actual Tprobe is
finite, but `T_mean_Rx_side_0_0` and `T_max_Rx_side_0_0` are absent. Seven use a
0.300 mm conductor and the other two use 0.327 and 0.440 mm conductors. The
campaign state independently records `singleton_rx_side_volume_missing`.
This is a thin exact-copper volume omitted from the Icepak cut-cell solution
mesh. A probe fallback cannot safely recover it because a partial group maximum
could understate the component hotspot.

Probe rectangles, not polylines, are used by the current extractor. There is no
current-cohort evidence that geometry variants produced an invalid Tprobe, that
reports were queried before solve validation, or that AEDT returned unit-tagged
or nonfinite values for otherwise valid current probes. There were nevertheless
three real fail-open diagnostic gaps:

- parameter-derived rectangles were not checked for finite coordinates or
  positive spans before crossing the AEDT boundary;
- a sheet creation failure could remove the sheet from `expected_cols`, hiding
  the original per-probe cause;
- missing Field Summary entities/statistics and conversion errors were silently
  discarded, leaving only NaNs in the result row.

The codebase contains a documented independent saved-field fallback in
`verify/replay_thermal_mesh.py`: `post.get_scalar_field_value("Temp", ...)`.
Historical live use showed that scalar calls do not recover zero-mesh volumes
and can be noisy over gRPC. The implemented fallback is consequently bounded to
missing probe *surfaces* after the existing three Field Summary attempts.

### `profile_mismatch:n_explicit_turns`

The 228 events are not a current producer/consumer mismatch. They are exactly
the historical nonzero population: 112 rows at 2, 77 at 4, and 39 at 1 explicit
turns. The standard profile now expects zero and the production thermal model
uses full-pack homogenized Rx blocks. All 228 predate the `thermal_rx_model` and
power-balance evidence; disabling profile matching recovers zero EM-valid or
strict-full rows.

The current path is aligned end to end:

- the standard profile and input default specify `n_explicit_turns=0`;
- scheduler profile overrides are applied after candidate values;
- `df_plus` carries that exact value into thermal geometry;
- zero selects `homogenized_blocks`, while nonzero selects `hybrid_explicit`;
- the quality contract checks both profile identity and the model echo.

The strict check remains in place. The hardening here explicitly pins zero in
the legacy campaign entry point and preserves both `n_explicit_turns` and
`thermal_rx_model` in train-I/O schema 6 so future audits cannot lose the
analysis basis.

## Implemented changes

### Probe extraction

- Validate probe name/orientation, three finite origin coordinates, and two
  finite positive spans before `create_rectangle`.
- Validate AEDT creation readback (non-bool object, exact name, `is3d=False`).
- Preserve every attempted probe name even when creation fails, so its required
  output columns remain missing and the row still quarantines.
- Parse finite numeric, Celsius-tagged, and Kelvin-tagged field values; reject
  unsupported/conflicting units and NaN/Inf.
- After three bulk Field Summary attempts, retry only missing required probe
  surface statistics through the scalar saved-field API. Missing volume fields
  never use this fallback.
- Emit `thermal_extraction_method`, `thermal_extraction_failure_reason`,
  `thermal_probe_failure_count`, and deterministic
  `thermal_probe_failures_json` telemetry. Unrecovered values remain NaN and
  `thermal_extraction_complete=0`.

### Explicit-turn contract

- Pin `--set n_explicit_turns=0` in `run_campaign.DEFAULT_ARGS`.
- Add `n_explicit_turns` and `thermal_rx_model` to the train-I/O analysis-basis
  columns and bump its schema from 5 to 6.
- Add tests proving standard profile override to zero, fine-profile override to
  two, profile mismatch quarantine for a historical hybrid row, and train-I/O
  preservation of the model basis.

## Expected yield impact

Quarantine reasons overlap, so the figures below are not additive.

| Category | Supplied snapshot | Current-revision diagnosis | Expected impact |
| --- | ---: | --- | --- |
| Thermal extraction required-missing | 380 | Nine converged cases, all singleton volume omissions | Probe fallback may prevent future Field Summary-only losses; zero of the nine known volume failures are reclassified |
| Tprobe/nonfinite and thermal flags | roughly 330-394 each | Mostly the 85 rejected solves, whose extraction is intentionally skipped | No contract relaxation; successful surface fallback can recover only genuine report fragility |
| Thermal convergence | roughly 305-367 each | 85 current rows, mainly iteration-ceiling residual failures | Unchanged in this work; must be measured in the pilot |
| `profile_mismatch:n_explicit_turns` | 228 | Historical nonzero physics, fully overlapping other stale-contract failures | Zero historical recovery; expected zero recurrence for standard submissions |
| Trusted current revision | n/a in mixed snapshot | 1,330 / 1,424 strict-full | 93.4% empirical pre-restart baseline, above the 85% target |

## Icepak mesh A/B verdict

`codex/icepak-mesh-ab-260712` commit `1071b3c` is one commit above `b171c7c`
and changes two files (+26/-9). Its functional mesh change lowers homogenized
`Rx_side_blocks` and `Rx_side2_blocks` from object mesh level 5 to 4. Singleton
exact Rx turns remain at level 5. It does not change convergence limits or the
iteration ceiling. The remaining change adds build/setup/solve/extraction
timing telemetry.

The mesh change was not cherry-picked. No `1071b3c` candidate result, mesh-zone
evidence, convergence result, temperature parity result, or timing result was
found in the supplied campaign data. The planned A/B baselines cover relatively
thick side packs (15.711, 48.675, and 51.315 mm), not the thinnest possible
multi-turn campaign pack. Historical evidence shows that insufficient
refinement can omit retained Rx zones. A level downgrade without A/B results
could reduce yield, and it does not address the nine observed singleton failures
because those already retain level 5. A three-way merge analysis found no text
conflict, but there is not yet evidence of benefit.

## Verification

Tests used the local `pyaedt2026v1` environment and the pinned external library
checkout at `Y:/git/pyaedt_library_worktrees/e6b9b9d/src`.

- `python -m py_compile` on every changed Python file: pass.
- Probe, thermal, campaign-entry, train-I/O, and the added strict-row contract
  tests: 54 passed, 8 subtests passed.
- Requested `regression_260707/test_simulation_stability.py` plus
  `regression_260707/test_al_integrity.py`: 203 passed, 58 subtests passed.
- `git diff --check b04b470..HEAD`: pass before the report commit.
- A broad run including all of `test_pipeline_completion.py` had 276 passes and
  four unrelated existing fine-finalization failures:
  `test_final_ranking_uses_fine_result_volume_not_stored_candidate_volume`,
  `test_fine_failure_falls_back_to_next_smallest_candidate`,
  `test_fine_uses_actual_geometry_and_exact_candidate_identity`, and
  `test_larger_unverified_candidate_does_not_block_smaller_fine_pass`. The only
  change in that file is the new explicit-turn test, which passes.

## Pilot-only validation

The supervisor's cluster pilot must validate:

- new-revision strict-full yield is at least 85%, with reason counts evaluated
  per row rather than added across overlapping histogram categories;
- AEDT accepts every validated probe sheet across the sampled geometry range;
- any forced Field Summary omission is recovered by the surface-only scalar
  fallback without extra solves, and failure JSON is present when it is not;
- the 250-iteration residual failure rate and monitor evidence rate;
- singleton Rx solution-zone preservation for 0.300-0.440 mm conductors;
- if the level-4 side-block experiment is reconsidered, mesh-zone preservation,
  convergence rate, runtime, and temperature parity on thin as well as thick
  multi-turn packs.
