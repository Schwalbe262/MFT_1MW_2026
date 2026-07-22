# Final1000 finite multi-seed Phase A implementation

Status: local implementation only; not published or authorized for Scheduler use.

Phase A reduces Scheduler attach/refill frequency by placing a finite ordered
seed block in one physical standard task. It deliberately starts a fresh
authenticated single-seed subprocess for every child. It does not reuse model
state and therefore does not claim any per-seed compute-time reduction.

## Implemented contract

- Batch protocol: `final1000-finite-multiseed-phase-a-v2`.
- Initial operational shape: one-child canaries followed by four-child
  canaries; the hard protocol bound is 1 through 8 children.
- Each child retains the authenticated current7 scientific identity, seed,
  bundle, stage profile, and resource identity. Phase A derives a new sealed
  execution envelope that adds `scheduler_cpus=4` and changes only
  `inference_threads` from the legacy profile value 8 to the allocated CPU
  count 4; reversing that transform must reproduce the validated v1 task.
- The immutable island profile continues to authenticate its eight-thread
  upper bound. The optimizer entrypoint accepts a lower runtime only when
  `scheduler_cpus == inference_threads <= profile_inference_threads`; remote
  preflight and terminal result both attest `scheduler_cpus=4` and
  `inference_threads=4`.
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` are all set from that
  same authenticated value. `optimizer_processes` remains 1, so a 4-CPU task
  cannot repeat the observed 8-thread `libgomp` oversubscription failure.
- The parent dedupe seals the protocol, immutable batch manifest, ordered child
  identities, bundle/stage, and the full resource policy including remote cwd.
- A lane runs exactly one child subprocess at a time. A terminal child receipt
  is created with `O_EXCL` before the next child can start.
- Seed-local authenticated optimizer failure advances to the next child.
  Missing/unreadable identity evidence and model-load/authentication failure are
  lane-fatal.
- SIGTERM, SIGINT, and the internal deadline latch permanently prohibit another
  child start. The current child gets a bounded terminate/kill grace path.
- A restarted parent refuses an existing journal. Previously sealed receipts
  remain harvestable, but unsealed seeds are not reused automatically.

Remote journal paths are:

- `runs/task-{parent}/batch_manifest.json`
- `runs/task-{parent}/task_status.json`
- `runs/task-{parent}/seed-{seed}/payload.json`
- `runs/task-{parent}/seed-{seed}/seed_status.json`
- the unchanged v1 result and 15 scientific artifacts below the same seed
  directory.

## Controller, harvest, and status seam

The controller-v3 transition layer upgrades an authenticated v1 ledger without
cancelling it, counts physical lanes against the exact 500 target and
200/160/90/50 stage quotas, reserves an entire child block atomically, and maps
both `timeout` and `timed_out` to a terminal state. It is intentionally a pure
transition module: it contains no HTTP client or apply/watch command.
For rolling migrations, every v1 entry stores the exact task envelope, source
launch-plan SHA, resource-policy identity, and envelope SHA authenticated from
its predecessor or successor cohort. Scheduler observation therefore never
reconstructs an old predecessor refill from a different successor template.

The production reconciliation driver is a separate layer. It defaults to
dry-run, is write-enabled only by explicit `--apply`, and has exactly one
Scheduler mutation endpoint: `POST /api/tasks`. It does not cancel or preempt
the predecessor. Its restartable watch loop persists a sealed state/history
chain after every cycle and reconciles a deterministic dedupe from the
latest-10,000 inventory after a POST-before-state-write crash.

The existing single-seed controller's read path now requests one namespace
inventory ordered by descending task ID per control cycle. Active IDs present
in that latest-10,000 window reconcile from the bulk response; only an active
ID omitted by the bounded window uses a task-detail fallback. Task ID, exact
name, and dedupe identity remain fail-closed.

The harvester authenticates the v2 parent against the exact bundle plan,
stable-reads its manifest and cursor, and reads only immutable receipts declared
by `sealed_child_count`. It can harvest those receipts while the physical parent
is still running. Mixed v1 terminal tasks and v2 parents retain real Scheduler
task IDs; no virtual task IDs are created.

The status adapter keeps at most 500 physical-lane summaries in the hot status.
Logical seed history is stored in immutable content-addressed shards. Frontend
hydration defaults to 256 records, rejects pages over 4096 records, and refuses
any normalized response at or above the deployed 32 MiB limit. Snapshot build
also proves that the largest individual record fits inside the full legacy
frontend wrapper. Hydration returns the largest contiguous requested prefix
that fits when a multi-record page would cross the cap.

## Required release gates

No task produced by this branch may be submitted against an older bundle. The
release preparation tool computes the full repository-local transitive import
closure from the optimizer preflight, seed runner, batch contract, and lane
runner. The old three-file extra-code list is not a complete bundle inventory
and must not be used as a release gate.

The bundle's authenticated optimizer entrypoint must be this commit's
`tools/tier1_corrected_generation_preflight.py`; it contains the runtime
CPU/profile/result validation. Reusing an older READY identity is prohibited.

The local release tool overlays every repository source already present in the
parent plus that computed closure, replaces `.source-revision`, performs an
isolated import smoke, and creates reproducible candidate/delta plans. It never
copies or creates READY. A later, separately authorized operation must publish
a new immutable bundle/READY identity and bind a new launch plan to it. Exact
backend/status/index/test capability must be deployed and sealed before any
batch task, because the deployed v1 reader assumes one Scheduler task per seed.

The rollout order remains: the old 500-lane controller keeps progressing; four
additive one-child lanes (one per stage, temporary physical maximum 504) pass;
then four additive four-child lanes (again one per stage, temporary maximum
504) pass. The phases reserve the upper five seeds of every stage (`end-5`,
then `end-4..end-1`) and never refill between gates. After both pass, the
driver waits for the exact final predecessor state and the canonical harvester
single-writer handoff. Only then may cancellation-free natural replacement
return to exactly 500 physical lanes and 200/160/90/50 stage quotas. Batch
length 8 and in-process model reuse remain separate changes requiring separate
evidence.

This implementation has not performed Scheduler HTTP mutation, cancellation,
preemption, remote publication, AEDT, or FEA. Production cutover remains
blocked until the mixed v1/v2 consumer handoff receipt is integrated and
tested.
