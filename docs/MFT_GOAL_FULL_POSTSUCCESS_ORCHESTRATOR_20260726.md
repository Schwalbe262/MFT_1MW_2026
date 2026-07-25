# Full terminal-success to final deliverable orchestrator

`tools/mft_goal_full_postsuccess_orchestrator.py` closes the read-only gap
after the first-truth Full fast lane has submitted exactly one Full task. It
does not submit, retry, reprioritize, cancel, delete, or otherwise mutate any
Scheduler object.

The handoff between the two independently deployable watchers is the sealed
`mft-goal-first-truth-full-fast-lane-receipt-v1` file. The post-success
watcher deliberately does not import the fast-lane module. It authenticates
the receipt seal and its exact Full submission/plan records, then uses the
existing strict truth-promotion loaders. This keeps integration order loose
without weakening the Full plan or submission authority.

## Release sequence

One watcher cycle performs the following fail-closed sequence:

1. Wait for the immutable fast-lane receipt. File absence is a normal waiting
   state.
2. Use Scheduler GETs to require exactly one task with the sealed Full task
   name and retained-artifact dedupe key. Its task ID must equal the strict
   Full submission receipt.
3. Wait while that task is active. A failed/cancelled/timed-out task is
   recorded as a terminal block and is never collected as successful truth.
4. On `completed`, reuse
   `mft_goal_truth_promotion.collect_full`. This reauthenticates the result,
   solver/library/profile/core identity, Full physics, all hard constraints,
   Full retained AEDT and results manifest, and the original Standard
   symmetric retained AEDT and results manifest.
5. Sum exactly these four authenticated receipt fields:

   - Standard `artifact_size_bytes`
   - Standard `results_size_bytes`
   - Full `artifact_size_bytes`
   - Full `results_size_bytes`

6. On the filesystem volume that will contain the final directory, require
   free bytes of:

   `exact retained bytes + ceil(exact retained bytes * 10 / 100) + 5 GiB`

7. Persist the immutable disk admission and an atomic pre-package intent.
   Only then call the existing atomic `package_results` implementation.
8. Run the complete `authenticate_package` validation. Then independently
   enumerate and SHA-256 hash every file in the package, rejecting missing or
   extra files and symlinks.
9. Emit `success_receipt.json` only after the complete validation and rehash.
   The success receipt, not final-directory existence, is the completion
   authority.

The stable deliverable directory is:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_deliverable_20260726`

Initialization requires its parent to exist but does not create this
directory. A pre-existing final directory is rejected on first
initialization. During recovery, a final directory without the sealed
pre-package intent is blocked rather than accepted.

The retained Standard and Full remote preservation markers are authenticated
by `collect_full` and remain required through package validation. This
orchestrator contains no remote prune operation.

## Prepare the GET-only watcher

The fast-lane output root may exist before its final receipt does. The final
deliverable root must not exist.

```powershell
$Python = "C:\Users\peets\anaconda3\envs\pyaedt2026v1\python.exe"
$FastRoot = "C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\first_truth_full_fast_lane_v1"
$PostRoot = "C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\full_postsuccess_v1"

& $Python tools\mft_goal_full_postsuccess_orchestrator.py init `
  --fast-lane-receipt "$FastRoot\fast_lane_receipt.json" `
  --output-root $PostRoot `
  --final-output-root "C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_deliverable_20260726" `
  --poll-seconds 15
```

This creates only operational watcher evidence under `$PostRoot`.

## Root-reviewed activation

This implementation was prepared without starting a live process. After code
review and integration, a root operator may run one GET-only cycle:

```powershell
& $Python tools\mft_goal_full_postsuccess_orchestrator.py once `
  --watch-plan "$PostRoot\watch_plan.json"
```

Or start the persistent GET-only watcher:

```powershell
& $Python tools\mft_goal_full_postsuccess_orchestrator.py run `
  --watch-plan "$PostRoot\watch_plan.json"
```

Neither command accepts a Scheduler activation receipt because neither has a
mutation path. The persisted state always records
`scheduler_methods_used=["GET"]` and false values for POST, PATCH, DELETE,
cancel, and generic mutation.

## Restart behavior

- Before Full terminal success, cycles only refresh GET observations.
- A valid existing `full_collection.json` is completely reauthenticated and
  reused.
- If packaging did not begin, a fresh same-volume disk reading creates a new
  immutable admission before the next attempt.
- `package_results` uses a sibling staging directory and an atomic directory
  rename.
- If the rename completed but the watcher stopped before writing success,
  restart requires the prior package intent, runs full package validation,
  rehashes every file, and only then creates the success receipt.
- A valid existing success receipt is still rechecked against a newly
  authenticated package and newly computed full-file hash inventory.
- Invalid or unexplained final-directory contents are never removed or
  overwritten automatically; the watcher fails closed for manual audit.

No branch, Scheduler project, or repository cleanup is part of this watcher.
