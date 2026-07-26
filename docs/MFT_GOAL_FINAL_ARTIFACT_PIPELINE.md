# Final AEDT and drawing artifact pipeline

## Scope

This is the post-selection pipeline for the 2026-07-26 MFT goal. It does not
choose a candidate and it does not turn Scheduler lifecycle success into a
scientific claim. It consumes one separately sealed, authenticated symmetric
hard-pass and prepares the following deliverables:

1. the exact retained symmetric, non-rounded verification AEDT;
2. a full, non-rounded AEDT built in `--model-only` mode;
3. a full, rounded-winding AEDT built in `--model-only` mode for drawings only;
4. a nine-slide PowerPoint and nine-page PDF based on
   `설계도면260706.pptx` and `설계도면260706.pdf`.

Rounded winding geometry is never a verification model and must never be
solved. The Full files in this pipeline are geometry/setup artifacts, not
substitutes for the authenticated symmetric scientific result.

## Fixed contracts

The winner authority and every derived parameter file fail closed unless all
of the following remain true:

- W is drawing x/original 973-mm direction and is at most 1200 mm;
- L is perpendicular drawing y and is at most 1000 mm;
- H is at most 750 mm;
- rotation and W/L axis exchange are forbidden;
- the primary conductor is 5.0 mm with 1.6 mm turn spacing;
- the total turns implement the 1:10 ratio;
- primary-referred Lm is 2 mH within 20 uH;
- minimum measured resonance is at least 15 kHz;
- primary winding is at most 100 °C;
- secondary winding is at most 120 °C;
- core is at most 120 °C;
- cooling is dual fan, 1.5 m/s;
- core and WCP pads are each 2.0 mm;
- thermal-pad conductivity/TIM is 0.2 W/(m·K);
- TIM is not mutated.

The Scheduler remains the separate `MFT_1MW_2026v1` project. This pipeline
contains no Scheduler source, POST, cancellation, project configuration
change, or branch creation.

## Current collection blocker

The active tuned symmetric retry is task 96743:

- name:
  `mft-final-sym-gap-2b2138a99445-g00860423-r1`;
- dedupe:
  `mft-al:mft-final-sym-gap-2b2138a99445-g00860423-r1:6af3e7e7cfba187b9711891c9457ce64835131b5:e6b9b9d20a832ff5c3f7ca97218737a0b8650781:0700f09bad77ebcc`;
- retained root: `goal-fea-retained/3c262e829b18d3a2`;
- AEDT: `goal-fea-retained/3c262e829b18d3a2/symmetric.aedt`;
- receipt:
  `goal-fea-retained/3c262e829b18d3a2/symmetric.aedt.receipt.json`;
- chunks:
  `goal-fea-retained/3c262e829b18d3a2/symmetric.aedt.chunks`;
- prune marker:
  `goal-fea-retained/3c262e829b18d3a2/.slurm-scheduler-preserve.json`.

There is currently no task-96743-specific sealed local submission plan and
receipt. The older files in
`rank1_neighborhood_2b213_physical_gap_tuning_v1` bind cancelled task 96483,
the old solver revision, the name without `-r1`, and retained token
`dd4d9287dd679b2a`; they cannot authorize task 96743.

Before task 96743 can feed this pipeline, a GET-only recovery collector must
seal all of the following:

1. exact task-96743 GET identity and terminal state;
2. exact Scheduler stdout and latest well-formed `RESULT_JSON`;
3. solver revision
   `6af3e7e7cfba187b9711891c9457ce64835131b5`;
4. library revision
   `e6b9b9d20a832ff5c3f7ca97218737a0b8650781`;
5. parameter digest `0700f09bad77ebcc`;
6. the source tuned-gap manifest and exact
   `symmetric_loss_thermal.json` parameter bytes;
7. the `goal_standard.json` profile identity;
8. remote AEDT receipt and prune marker;
9. every contiguous base64 chunk, its hash and size;
10. reconstructed AEDT hash and size equal to the remote receipt;
11. measured matrix/cap/loss/thermal gates and the split temperature limits.

The raw remote AEDT or reconstructed chunks alone are not winner authority.
Operational failure also is not scientific infeasibility.

## Winner authority input

`tools/mft_goal_final_artifact_pipeline.py` accepts a sealed
`mft-goal-final-artifact-winner-authority-v1` JSON file. The upstream
collector/selector, not this tool, must create it. Its required topology is:

```text
schema_version
source
  task_id
  candidate_physics_sha256
  selection_performed_upstream = true
  candidate_promotion_performed_by_pipeline = false
  selected_symmetric_hard_pass = true
  upstream_selection_receipt = {path, sha256, size_bytes}
  retained_symmetric_aedt = {path, sha256, size_bytes}
contracts
  goal_contract_schema
  goal_stage_spec_sha256
  goal_temperature_contract_sha256
  fixed_operating_identity_sha256
  fixed_cooling_identity_sha256
  fixed_boundary_contract_sha256
params
  exact symmetric non-rounded Matrix+Cap+Loss+Thermal parameters
symmetric_verification
  full_model = 0
  round_corner = 0
  matrix_solved/capacitance_solved/loss_solved/thermal_solved = true
  measured_hard_constraints_passed = true
  rounded_fea_used = false
  actual_dimensions_mm = {W, L, H}
  actual_resonance_Hz
  actual_Lm_primary_referred_H
  actual_temperature_family_max_C
  fixed_boundary
payload_sha256
```

All file records and the canonical payload seal are revalidated before any
derived files are written.

## Preparation command

Run with the AEDT Python environment because strict parameter validation uses
the production input module:

```powershell
& 'C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe' `
  tools\mft_goal_final_artifact_pipeline.py `
  --winner-authority <sealed-winner-authority.json> `
  --output <final-artifact-workspace>
```

This creates only:

- `pipeline_manifest.json`;
- `params/full_final_nonrounded.model_only.json`;
- `params/full_final_rounded_drawing_only.model_only.json`.

The sealed manifest contains exact argv, working directories, dependencies,
postconditions, output names, and QA gates. Preparation does not launch AEDT,
solve, copy the source symmetric file, or edit the reference drawings.

## Parallel execution after sealing a winner

Wave 1 can start in parallel:

- copy the authenticated symmetric AEDT byte-for-byte to
  `models/final_symmetric_nonrounded.aedt` and rehash it;
- run the Full non-rounded command from the manifest in `--model-only` mode;
- run the Full rounded command from the manifest in `--model-only` mode;
- prepare the candidate specification table and nine-slide edit map.

Wave 2 starts after the rounded model exists:

- export five drawing views with
  `tools/mft_goal_export_rounded_drawing_views.py`;
- open both Full models for geometry/readback QA without a solve.

Wave 3 duplicates all nine reference slides and edits inherited elements with
`@oai/artifact-tool`. Rebuilding the deck from a blank presentation is
forbidden.

Wave 4 runs in parallel:

- render and inspect all nine PPTX slides;
- export, render at 240 dpi or higher, and inspect all nine PDF pages;
- seal SHA/size records for all three AEDT files.

## Drawing reference evidence

The source files are immutable:

- PDF SHA256:
  `574d9aab033529cf3655d63542e27871c2e240b669b67e54dbfd2495a564437f`;
- PPTX SHA256:
  `b8069cc99cf1af3c8e5c5cbe6bd4a2896730620f570c29555ff492713d4af33d`.

The completed reference audit is:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\drawing_reference_audit_v1`

Every one of the nine PPTX slides and nine PDF pages was visually inspected.
The edit mapping preserves:

1. cover;
2. overall top;
3. overall front;
4. core top;
5. core front;
6. center winding;
7. side winding;
8. winding separation;
9. conductor/pitch.

The rounded winding is a racetrack/rounded rectangle. Drawings must dimension
x/y straight spans, corner radius/radius ladder, conductor cross-section,
radial build/gaps, z height/pitch, and core/WCP/pad clearances. Circular ID,
OD, or mean-turn diameter notation is forbidden unless the actual geometry is
circular.

The prior artifact-tool source render did not show the cover research-lab
subtitle and converted some dashed dimension extensions to solid lines.
Therefore final fidelity must be checked against the source PDF or native
PowerPoint render, not only the artifact-tool raster.

## Artifact-tool runtime preflight

Version 2.8.31 was successfully linked into a fresh workspace. On Windows the
setup helper falls back to its current working directory when no `HOME`
environment variable is supplied, so invoke it from `C:\Users\peets` without
changing `HOME`:

```powershell
Push-Location 'C:\Users\peets'
node `
  'C:\Users\peets\.codex\plugins\cache\openai-primary-runtime\presentations\26.723.12215\skills\presentations\container_tools\setup_artifact_tool_workspace.mjs' `
  --workspace <final-artifact-workspace\drawings\artifact_tool_workspace>
Pop-Location
```

Final authoring still must use the imported source presentation, duplicate all
nine slides, preserve the native master/layout topology, render every slide,
and pass overflow/template-fidelity checks.
