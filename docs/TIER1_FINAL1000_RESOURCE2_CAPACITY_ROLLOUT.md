# Final1000 entry-stage 2-CPU capacity rollout

This rollout is deliberately narrower than an all-stage 2-CPU promotion.
Task `86501` exercises only `entry-1200-t125`; therefore only new entry-stage
refills may use 2 CPUs. Bridge, close, and final continue to use the existing
4-CPU task contract until each has independent terminal evidence. Existing
tasks are never cancelled or preempted.

## Sealed identities

- Running predecessor plan: `90af982bf32b8fb005b8557db77f1822f5cc8a2c3dc6e377c29da41e4cdfe730`
- Running predecessor code: `4aaf99a02f54d134dfe202b2867a73bab1460577`
- Mandatory canary: task `86501`
- Canary package: `12f8ca284d4cf3d12f8bc9e6ac528473e9872e8e81300849dfc1ae03a6e9a7b8`
- Canary submission: `da0bcbde3c626ce2b9f003746b8f0bb5b6e4f2130edc54ac6bc8123545bdd742`
- Rollout config: `docs/evidence/tier1_final1000_resource2_capacity_rollout_20260723.json`
- Rollout config SHA: `f407a7847925f15ac188c315fa17b4422fda3cf7056a530156ec3a02d4c1f70a`

At the observed 41 active CPU allocations (1,886 CPUs), the exact
allocation-local fits are 459 all-4-CPU tasks, 719 all-2-CPU tasks, and 574
tasks for the gated mixed policy. The mixed target is apportioned as
230/184/103/57; only the 230 entry slots use 2 CPUs. The controller recomputes
this integer bin-pack every cycle, excludes draining/pending/warm allocations,
enforces a logical floor of 500, and caps growth at 1,000.

## Mandatory terminal gate

Do not stop or replace either live process until Scheduler GET reports task
86501 terminal. A non-completed terminal state is a hard refusal. Once it is
completed, create the GET-only terminal evidence with the immutable inputs:

```powershell
$Source = 'C:\w\mftresource2rollout'
$Prep = 'C:\w\resource2_canary_prep_260723'
$Baseline = 'C:\w\phaseb_shape1_retry_pbd6_entry'
$Publication = 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721\topology_niche_successor_candidate_20260722\phaseb-topology-runtime-bee2111-publication-v2\entry-1200-t125-publication-receipt.json'

python "$Source\tools\tier1_final1000_resource2_canary.py" evaluate-remote `
  --config "$Source\docs\evidence\tier1_final1000_resource2_canary_v5_20260723.json" `
  --offload-plan 'C:\w\pbd6\e\current7-5f6ae36bafad7db9bd2e\offload_plan.json' `
  --publication-receipt $Publication `
  --baseline-package "$Baseline\shape1_retry_package.json" `
  --baseline-submission "$Baseline\shape1_retry_submission.json" `
  --baseline-terminal "$Baseline\shape1_retry_terminal.json" `
  --package "$Prep\resource2_package_v5.json" `
  --submission-receipt "$Prep\resource2_submission_v5.json" `
  --task-id 86501 `
  --scheduler-url http://127.0.0.1:8002 `
  --out "$Prep\resource2_terminal_v5.json"
```

The rollout validator must report `promotion_eligible=true`,
`scheduler_status=completed`, GET-only access, zero Scheduler POST/cancel/
preempt, and the exact task/package/submission identities above. There is no
override flag. The nested terminal schema does not carry a submission-receipt
field, so submission identity is bound by the outer remote-evidence hash and
then matched to the sealed rollout config. Task, package, status, and terminal
SHA are additionally cross-checked between the outer wrapper and nested
terminal. The exact 13-gate inventory is required and every value must be the
JSON boolean `true`; missing, extra, false, or merely truthy values fail closed.

## Dry-run before handoff

Use an immutable release path for the committed code. The live predecessor is
allowed to remain running for this dry-run; no state file or Scheduler POST is
created without `--apply`.

```powershell
$Release = '<IMMUTABLE_ROLLOUT_RELEASE>'
$Runtime = 'C:\Users\peets\slurm_scheduler_runtime\mft_tier1_final_goal_1000_t100_resmax20_260721'
$Plan = "$Runtime\multiseed_release_60e59f4_adapterbound_260722_1856\launch-plan-batch4-500.json"
$OldState = "$Runtime\controller\state-batch4-90af982b-external-canary-4aaf99a.json"
$Config = "$Release\docs\evidence\tier1_final1000_resource2_capacity_rollout_20260723.json"
$Terminal = 'C:\w\resource2_canary_prep_260723\resource2_terminal_v5.json'
$NewState = "$Runtime\controller\state-resource2-entry-capacity-f407a784.json"

python "$Release\tools\tier1_final1000_resource2_capacity_rollout.py" prepare `
  --plan $Plan --predecessor-state $OldState --state $NewState `
  --config $Config --resource2-terminal $Terminal `
  --scheduler-url http://127.0.0.1:8002
```

Confirm the printed target and allocation count. Re-run if the active pool
changed. The production prepare step recomputes capacity again.

## Atomic handoff

1. Signal the existing controller stop file and wait for PID 54792 to exit.
   This seals `stop_requested=true`; it does not cancel any task.
2. Signal the existing harvester stop file and wait for PID 63152 to exit.
3. Run `prepare --apply` once. It requires the authenticated predecessor state
   to be stopped and writes only the new local state.
4. Start the new controller. It retains every existing 4-CPU task and fills the
   current mixed target. With the 41-allocation snapshot this means 74 new
   tasks: 30 entry tasks at 2 CPUs and 44 other-stage tasks at 4 CPUs.
5. Start the compatible harvester against the new state, adding the rollout
   config and terminal gate arguments.

```powershell
python "$Release\tools\tier1_final1000_resource2_capacity_rollout.py" prepare `
  --plan $Plan --predecessor-state $OldState --state $NewState `
  --config $Config --resource2-terminal $Terminal `
  --scheduler-url http://127.0.0.1:8002 --apply

python -B "$Release\tools\tier1_final1000_resource2_capacity_rollout.py" run `
  --plan $Plan --state $NewState --config $Config `
  --resource2-terminal $Terminal --scheduler-url http://127.0.0.1:8002 `
  --accounts 'Y:\runtime\slurm_scheduler\config\accounts.yaml' `
  --scheduler-source 'C:\Users\peets\NEC\slurm_scheduler' `
  --publication-account harry261 --apply --watch --poll-seconds 15 `
  --stop-file "$Runtime\STOP_FINAL1000_RESOURCE2_ENTRY_F407A784"
```

Restart the harvester with its existing launch/binding/predecessor/ancestor
arguments unchanged, replace `--controller-state` with `$NewState`, use the
new release script, and append:

```text
--resource2-rollout-config <...capacity_rollout_20260723.json>
--resource2-terminal-gate C:\w\resource2_canary_prep_260723\resource2_terminal_v5.json
```

## Failure behavior

The first non-completed terminal result from any new 2-CPU task atomically
switches future refills to the sealed 4-CPU policy and lowers the target to
500. Existing 2-CPU and 4-CPU tasks continue naturally. No cancellation,
preemption, FEA, AEDT, allocation, service, or database mutation exists in the
controller. Refill reservations are persisted and posted one at a time. A
first 2-CPU terminal failure is persisted together with fallback before any
later reservation or POST; a crash after POST but before the response state is
persisted reconciles the same exact dedupe on restart. At most one unposted
reservation can exist, and fallback either authenticates an already-created
task or releases that reservation and rewinds its seed without a POST. A
manual fallback is available only after stopping the controller and requires
the exact current state SHA:

```powershell
python "$Release\tools\tier1_final1000_resource2_capacity_rollout.py" fallback `
  --plan $Plan --state $NewState --config $Config `
  --resource2-terminal $Terminal --expected-state-sha256 '<CURRENT_STATE_SHA>' `
  --apply
```

## Later all-stage expansion

Do not edit `resource2_promoted_stage_ids` in this v1 config. Bridge, close,
and final need independent 2-CPU canaries with terminal CPU/RSS/throughput
evidence. Run those three canaries in parallel. A later sealed config/state
migration may promote each stage only after its own gate; once all four gates
pass, the same 41-allocation pool supports the observed all-2-CPU target of
719. Different pool shapes are always recalculated rather than hard-coded.
