# Main consolidation record — 2026-07-28

## Outcome

The validated integration line was promoted to `main` with a non-force
fast-forward push.

- Previous `origin/main`: `4bdf44ae115a38228d0d8bd90806f55f76b150bf`
- Integration baseline: `8589827f40855af2f9e292736b2fb92212385332`
- Validated code tip: `02afd14897c8f423b8e34250bd024919fddfe1c3`
- Initial remote promotion: `4bdf44a..02afd14`
- Clean canonical worktree: `C:\w\mft-main-consolidation-20260728`
- Preserved dirty root worktree: detached at its existing HEAD; files were not
  reset, deleted, or overwritten

## Included changes

The consolidation includes the reviewed integration history plus narrowly
scoped fixes found by the final parallel validation:

- safe H390 visible model-only launcher; it cannot dispatch analysis or close
  an existing GUI;
- dynamic retry authority using the sealed attempt counts actually present;
- split primary, secondary, and core temperature limits across search,
  collectors, selection watches, and legacy aggregate compatibility;
- fixed three-leg air-gap test fixtures and strict equal-gap compatibility;
- monotonic stage timing and Starlette 0.41 test-client compatibility;
- deterministic thermal storage capacity tests;
- exact Git reads from linked worktrees on filesystems without ownership
  metadata;
- fail-closed authentication of explicitly identified historical sealed
  scheduler commands, launchers, B3 mesh evidence, and clean-checkout Git
  blobs;
- a replay-only legacy rounded-correction repair pinned to its reviewed source
  revision. It is marked `launch_eligible=False`; the production 40 mm axial
  policy remains unchanged.

## Deliberately excluded

The following dirty or provisional changes were not promoted:

- broad unreviewed thermal-core rewrites;
- unsealed resume/pass-7 provenance;
- hard-coded final-authority or drawing promotion claims;
- refill/rollover accounting that was not backed by live Scheduler evidence;
- provisional core-rescue changes;
- generated solver output, AEDT files, caches, `node_modules`, virtual
  environments, and temporary result trees.

Those files remain recoverable from the dirty-worktree archive described
below. Large simulation and drawing directories were preserved in place.

## Verification

The source was held at `02afd14` for the final validation.

- Top-level `tests`: 155 files, 1,956 passed, 1 skipped, 0 failed
- `regression_260707`: 921 passed, 48 skipped, 0 failed
- Combined: 2,877 passed, 49 skipped, 0 failed
- Collection audit: 2,926 tests collected
- Test stderr files: 0 non-empty
- Changed Python files from the integration baseline: 34 compiled
- `git diff --check 8589827..02afd14`: passed
- Final logs:
  `C:\w\mft-git-archives\mcb-20260728_172028\final-validation-02afd14`

The older integration history contains whitespace in a few retained raw
experiment artifacts, so `git diff --check 4bdf44a..02afd14` reports those
historical artifact lines. No consolidation commit introduced new whitespace
errors.

## Backup and recovery

- Full pre-cleanup bundle:
  `C:\w\mft-git-archives\MFT_1MW_2026_pre_cleanup_20260728_152645.bundle`
  - SHA256:
    `fd5d03d434834b6f31dc688d481ec3c54488523dd7f05d5d698079d604089668`
- Dirty-worktree archive:
  `C:\w\mft-git-archives\mcb-20260728_172028`
  - 23 dirty worktrees
  - 525 manifest records
  - manifest SHA256:
    `4024d8aabd45eb10109557940cc84773f48683d98857022fada291afd3e06d47`
  - rehash mismatches: 0
- Immediate pre-main-cleanup bundle:
  `C:\w\mft-git-archives\MFT_1MW_2026_pre_main_cleanup_20260728_182811.bundle`
  - SHA256:
    `0bd0d1588999143aab610e36dce51db44e958558625a24186b262c6bf4716b36`
  - `git bundle verify`: complete history, 663 refs

## Local branch and worktree cleanup

The earlier cleanup reduced local branches from 316 to 31. Seven temporary
integration branches were then created for this consolidation, producing 38
branches and 550 registered worktrees at the final cleanup boundary.

- 29 non-main attached worktrees were moved to the same detached HEAD without
  changing their files;
- 37 bundle-protected local branches were deleted;
- final local branches: 1 (`main`);
- final attached worktrees: 1 (`main`);
- six clean `C:\w\mft-main-*` helper worktrees were removed;
- final registered worktrees: 544 (1 attached, 543 detached);
- remote feature branches were not deleted or rewritten;
- data-bearing worktree directories, including simulation and drawing
  results, were not removed.

## Physical-design authority

This Git consolidation validates code behavior, not the final transformer
physics.

- Global NDS hard-feasible count remains 0.
- The symmetric thermal authority is not a valid PASS.
- The full-curved AEDT is model-only.
- Existing drawing and electrical artifacts remain provisional until the
  required symmetric thermal/FEA authority succeeds.

The 8010 UI must continue to show this distinction instead of describing the
physical design as fully validated.
