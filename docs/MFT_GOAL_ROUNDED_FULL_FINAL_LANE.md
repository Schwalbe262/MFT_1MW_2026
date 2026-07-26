# Rounded candidate-5 single Full cross-check lane

This lane is prepared for the exact rounded candidate-5 Standard task 96340.
It is not an automatic Standard-to-Full continuation and it cannot select a
different candidate.

## Hard gates

`prepare` fails closed unless all of the following are true:

- task 96340 is terminal `completed/succeeded`, exit code 0, on strict
  `dw16/n113`;
- its authenticated GET-only collection contains the real retained
  `symmetric.aedt`;
- its measured dimensions, resonance, and every active winding/core
  temperature target pass;
- its fixed thermal identity is still dual 1.5 m/s fans, 0.2 W/(m K)
  TIM/insulation, 2 mm core pads, and 2 mm WCP pads;
- the Full profile keeps `round_corner=1`, radius 10 mm, four chords per
  quarter corner, and changes only to `full_model=1`,
  `thermal_symmetry=full`;
- a fresh 16-core license snapshot is at most 180 seconds old; and
- the selected strict Full node is different from n113, FEA-empty, and has at
  least 16 CPUs and 98,304 MB available.

The solve envelope is 79,200 seconds inside one 86,400-second Scheduler task.
The retained terminal bundle is `full_model.aedt`,
`full_model.aedtresults`, and the authenticated results-tree manifest.

## Timeout-safe diagnostic checkpoint

The terminal Full retention path runs only after solver success, so the lane
adds a separate pre-solve checkpoint:

1. run the exact rounded Full payload once with `--model-only`;
2. after AEDT saves geometry and Setup, atomically copy the single project;
3. hash it, split it into bounded base64 chunks, and write a separate
   diagnostic receipt and preserve marker under `goal-fea-checkpoints/`;
4. remove the temporary model-only project, refresh the runtime 16-core
   license evidence a second time, then start the bounded Full solve.

This checkpoint remains GET-recoverable when the later Full task fails or
times out. It is permanently marked:

- `scientific_pass=false`
- `thermal_pass=false`
- `production_promotion_eligible=false`
- `diagnostic_only=true`

Only a terminal successful `RESULT_JSON` and the terminal Full retained
bundle may be used for the Full scientific verdict. The checkpoint can supply
a rounded geometry/setup AEDT file, never a thermal result or promotion.

## Commands

The completed no-I/O dry run is:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_official5_rounded_full_prepare.py dry-run `
  --standard-plan C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_standard_official5_rounded_r10_s4_prepare_v3\rounded_final_prepare_plan.json `
  --strict-lane dhj02=n116 `
  --license-snapshot <fresh-snapshot.json> `
  --output C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\rounded_full_dry_run_v5.json
```

Dry-run v5 performed zero Scheduler GETs and zero POSTs. Its sealed payload
SHA-256 is
`66b281e8b65ff9f1b6855ad6b4391a89bb801d99cce549cc686d49dbeeb93b45`.
`bash -n` accepted the generated remote command.

After task 96340 succeeds, select a currently FEA-empty strict lane and run:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_official5_rounded_full_prepare.py prepare `
  --standard-plan <task96340-rounded-plan.json> `
  --standard-final <task96340-submission-final.json> `
  --standard-collection <task96340-authenticated-collection-directory> `
  --strict-lane <ACCOUNT=NODE> `
  --license-snapshot <fresh-snapshot.json> `
  --output-root C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_full_official5_rounded_r10_s4_prepare_v1
```

That command is GET-only. Review its sealed plan before the only authorized
POST:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_official5_rounded_full_submit.py `
  --plan C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_full_official5_rounded_r10_s4_prepare_v1\rounded_full_prepare_plan.json `
  --authorize-post authorize-rounded-full-one-post-v1
```

Do not run the submit command before task 96340 passes. Its immutable attempt
ledger is written before the network call; an ambiguous attempt is never
re-posted.

GET-only collection is:

```powershell
C:\Users\peets\anaconda3\python.exe `
  tools\mft_goal_official5_rounded_full_collector.py `
  --plan <rounded-full-prepare-plan.json> `
  --submission <rounded-full-submission-receipt.json> `
  --output <authenticated-full-collection-directory>
```

On `failed`, `cancelled`, or `timed_out`, the same collector attempts the
diagnostic checkpoint fallback and records that it is not promotion eligible.
No tool in this lane edits or imports the separate Scheduler repository.
