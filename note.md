# Loop Notes

This file is append-only. Each RL loop appends timestamped execution status, batch size, backend, job IDs, completion counts, best reward, and errors.

## Loop 1 - 2026-06-15T03:23:59Z

- backend: local-mock
- status: completed
- batch_size: 3
- completed: 3
- failed: 0
- job_ids: -
- best_reward: 37.9407582245
- best_candidate_id: L0001-C0002
- finished_at: 2026-06-15T03:23:59Z

## Loop 2 - 2026-06-15T03:30:05Z

- backend: local-mock
- status: completed
- batch_size: 4
- completed: 4
- failed: 0
- job_ids: -
- best_reward: 44.983672651
- best_candidate_id: L0002-C0003
- finished_at: 2026-06-15T03:30:05Z

## Loop 3 - 2026-06-15T03:43:39Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 0
- failed: 1
- job_ids: 1
- best_reward: -
- best_candidate_id: -
- finished_at: 2026-06-15T03:44:39Z
- message: Slurm jobs finished; fetched shared result file: 0/1.

## Loop 4 - 2026-06-15T03:46:56Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 0
- failed: 1
- job_ids: 2
- best_reward: -
- best_candidate_id: -
- finished_at: 2026-06-15T03:48:56Z
- message: Slurm jobs finished; fetched shared result file: 0/1.

## Loop 5 - 2026-06-15T03:49:51Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 3
- best_reward: -
- best_candidate_id: -
- finished_at: 2026-06-15T03:51:50Z
- message: Slurm jobs finished; fetched shared result file: 0/1.

## Loop 6 - 2026-06-15T03:52:47Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 4
- best_reward: 3449.51650441
- best_candidate_id: L0006-C0001
- finished_at: 2026-06-15T04:02:39Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live baseline established.

## Loop 7 - 2026-06-15T04:02:48Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 5
- best_reward: 3105.67933287
- best_candidate_id: L0007-C0001
- finished_at: 2026-06-15T04:14:37Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: False.

## Loop 8 - 2026-06-15T04:14:45Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 6
- best_reward: 3087.35190486
- best_candidate_id: L0008-C0001
- finished_at: 2026-06-15T04:27:32Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: False.

## Loop 9 - 2026-06-15T04:27:41Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 7
- best_reward: 3190.45605896
- best_candidate_id: L0009-C0001
- finished_at: 2026-06-15T04:37:31Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: False.

## Loop 10 - 2026-06-15T04:37:39Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 0
- failed: 1
- job_ids: 8
- best_reward: -
- best_candidate_id: -
- finished_at: 2026-06-15T04:40:38Z
- message: Slurm job was cancelled before execution; fetched stale loop 9 result was discarded after candidate-id filter fix.

## Loop 11 - 2026-06-15T04:42:13Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 9
- best_reward: 3397.16281355
- best_candidate_id: L0011-C0001
- finished_at: 2026-06-15T04:54:59Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: False.

## Loop 12 - 2026-06-15T04:55:17Z

- backend: slurm
- status: completed
- batch_size: 1
- completed: 1
- failed: 0
- job_ids: 10
- best_reward: 3445.39546616
- best_candidate_id: L0012-C0001
- finished_at: 2026-06-15T05:05:06Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: False.

## Loop 13 - 2026-06-15T05:05:53Z

- backend: slurm
- status: completed
- batch_size: 20
- completed: 0
- failed: 20
- job_ids: 11
- best_reward: -
- best_candidate_id: -
- finished_at: 2026-06-15T05:07:52Z
- message: Slurm batch was cancelled while pending because 20 workers * 4 CPUs exceeded QOSMaxCpuPerNode; stale loop 12 result was discarded.

## Loop 14 - 2026-06-15T05:08:27Z

- backend: slurm
- status: completed
- batch_size: 20
- completed: 11
- failed: 9
- job_ids: 12
- best_reward: 3631.97429028
- best_candidate_id: L0014-C0007
- best_outputs: Ltx=2287.3392632692703, Lrx=227422.93085759002, M=22712.956695389403, k=0.99584386999495, Lmt=2268.3658147457104, Lmr=225536.460695076, Llt=18.9734485235606, Llr=1886.4701625138002, Tx_loss=2274.8257335807402, Rx_loss=306.170300864985, time=698.4342918395996
- finished_at: 2026-06-15T05:26:07Z
- message: Slurm jobs finished; fetched shared result file: 1/1. Live improved: True. Partial batch retained 11/20 result rows after cancelling stalled workers.
