# Rounded B5 symmetric n107 hedge evidence

Status captured at 2026-07-27 02:27 KST. This record documents the one
authorized hedge submission and its current admission blocker; it is not a
second submission authorization.

## Immutable submission

- Scheduler task: `96342`
- Created: `2026-07-26 17:17:03 UTC` (`2026-07-27 02:17:03 KST`)
- Name:
  `mft-goal-final-standard-official5-rounded-r10-s4-hedge1-v1-909d249ebe45-n107`
- Placement: requested account `harry261`, strict node `n107`, fresh
  allocation only, `max_workers_per_node=1`
- Physics: candidate SHA-256
  `909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42`;
  rounded winding `R10/s4`; eighth symmetry; `full_model=0`; dual fan at
  `1.5 m/s`; `k_ins=0.2`; 2 mm core and winding cooling-plate pads
- POST receipt: exactly one Scheduler POST, consumed before the network call;
  no automatic retry is allowed
- Final seal:
  `C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_standard_official5_rounded_r10_s4_hedge_n107_v1\submission\final_seal.json`

The GET-only collector remains active and writes:

`C:\Users\peets\slurm_scheduler_runtime\mft_goal_20260726\final_standard_official5_rounded_r10_s4_hedge_n107_v1\authenticated_get_collection\latest.json`

At capture time the authenticated readback was `queued`, with no allocation
or Slurm job and `new_allocation_authenticated=false`. The collector performs
no mutation and records `scheduler_post_calls=0`.

## Hidden storage-admission blocker

The public capacity endpoint reported an opening lane, but the AEDT pool
status and Scheduler log show that `harry261` is blocked by the account
storage guard:

- GPFS block quota: 98.9 / 110.0 GB used plus in-doubt
- Observed free: 11.1 GB
- Prospective AEDT reservation: 12.0 GB
- Effective free: -0.9 GB
- Required floor: 10.0 GB

The 12 GB is deliberate, not stale state. The pool checks one prospective
session with three project slots, and the configured reservation is 4 GB per
slot. Starting a session therefore requires approximately 10.9 GB more
observed quota headroom. The warning repeated hourly through
`2026-07-27 02:24:29 KST`.

A read-only remote audit found about 0.77 GiB in six July 15
`~/slurm_scheduler/runs/mft_campaign-*` scratch directories. They had no
preserve markers or open handles, but reclaiming all of them would still be
far short of 10.9 GB. The remaining large holdings belong to unrelated
projects or environments, so no file was deleted, moved, or modified.

## Safe operational rule

Do not submit this hedge again. Continue GET collection. To make task 96342
start, either free at least about 10.9 GB within the exact `harry261` GPFS
quota/fileset or place an explicitly authorized replacement on an account
whose storage guard passes. Neither action is authorized by this evidence
record.
