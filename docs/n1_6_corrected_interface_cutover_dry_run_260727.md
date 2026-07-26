# N1=6 corrected-interface cutover dry run

Status: plan only. No cancellation or replacement submission is authorized by
this document.

## Scope and invariants

- Scheduler origin remains `http://127.0.0.1:8002`.
- Scheduler project remains `MFT_1MW_2026v1`.
- The Slurm Scheduler repository, service, allocations, node policy, and
  configuration are not modified.
- Physics remain one-eighth, non-rounded symmetric FEA with dual 1.5 m/s fan,
  2 mm TIM pads, and TIM conductivity 0.2 W/(m*K).
- A corrected result must emit all six explicit Rx-interface/limiter fields and
  pass the acquisition collector's scientific truth gate.
- Replacement tasks are accepted by Scheduler before any old task is
  cancelled. This prevents an empty compute interval.

## Existing tasks affected

Old N1=8, monitor/forensic only; cancel without replacement at cutover:

- `96743`

N1=6 cooler hedge, replace one-for-one:

| Rank | Old task | Geometry prefix |
|---:|---:|---|
| 1 | 97014 | 656b673929bb |
| 2 | 97015 | 54f8195b9291 |
| 3 | 97016 | 69265bf48804 |
| 4 | 97017 | 1d7d7aa18ce9 |
| 5 | 97018 | 7ea9e7726e24 |
| 6 | 97019 | ef307f9a6d15 |
| 7 | 97020 | 8b6cbb06edc8 |
| 8 | 97021 | 5f1aa606edad |
| 9 | 97022 | dfa1de277bd8 |
| 10 | 97023 | d688cb2f97ab |
| 11 | 97024 | fbbdf1ed9e74 |
| 12 | 97025 | cf16e62f6f8f |
| 13 | 97026 | 5ca80ac0528c |
| 14 | 97027 | 0daa658bf10a |
| 15 | 97028 | 0bad6363e2f3 |
| 16 | 97029 | 941bbe696295 |

N1=6 primary-temperature last-mile, replace one-for-one:

| Rank | Old task | Geometry prefix |
|---:|---:|---|
| 1 | 97033 | d03866b78823 |
| 2 | 97034 | 0fbcc8e73970 |
| 3 | 97035 | 690ef78d4c97 |
| 4 | 97036 | 26033d7b5cf9 |
| 5 | 97037 | b31c08a13f3a |
| 6 | 97038 | db7c5d4e8be2 |
| 7 | 97039 | 936854df051b |
| 8 | 97040 | 5e64db5d7be5 |

The exact old-task cancellation set is:

`96743,97014,97015,97016,97017,97018,97019,97020,97021,97022,97023,97024,97025,97026,97027,97028,97029,97033,97034,97035,97036,97037,97038,97039,97040`

## Transaction gates

1. Corrected canary is Scheduler-accepted and attached to a real allocation.
   Its task readback must match the corrected solver/library revisions,
   project, dedupe key, 8 CPU, 65536 MiB, and standalone AEDT backend.
2. Canary stdout must show the corrected Rx-interface implementation identity
   and no unpaired-interface-to-wall or 5000 K limiter marker. Prefer a
   terminal six-field scientific-valid result; a pre-solve deterministic
   interface attestation is acceptable only with explicit root authorization.
3. Two immutable corrected replacement plans contain exactly the 24 N1=6
   geometry hashes listed above, with unchanged effective physics parameters.
   `96743` is deliberately not replaced.
4. Replacement submission receipts are complete and sealed. Every replacement
   task has a unique new task ID and dedupe key, and every GET readback is
   `queued`, `attaching`, or `running`.
5. Only after gates 1--4 pass is cancellation allowed.

If any gate fails, do not cancel an old task. Preserve both old campaigns and
repair the replacement plan first.

## Cutover sequence

1. Snapshot all 25 old task readbacks and logs.
2. Submit all 24 corrected N1=6 replacements in parallel. Keep the cooler
   campaign at priority 95 and the last-mile campaign at priority 90 unless the
   root operator explicitly seals a different priority. The corrected canary
   remains priority 100.
3. Authenticate the replacement task-ID set, geometry set, revisions, fixed
   boundaries, and accepted/attached statuses.
4. Immediately issue one conditional bulk cancellation against only the exact
   old-task set:

   `POST /api/tasks/cancel?task_ids=96743,97014,97015,97016,97017,97018,97019,97020,97021,97022,97023,97024,97025,97026,97027,97028,97029,97033,97034,97035,97036,97037,97038,97039,97040&statuses=queued,attaching,running`

5. Re-read all old and new tasks. A status-race old task that became terminal is
   collected as forensic evidence and remains scientific-invalid. Never blind
   cancel a terminal task.
6. Verify remote wrappers of cancelled tasks are being reclaimed and the 24
   corrected tasks remain queued/attaching/running. Do not edit Scheduler
   configuration to force placement.
7. Monitor the corrected tasks with the six-field fail-close collector. The
   first authenticated N1=6 split-temperature PASS immediately starts the
   priority-100 gap bracket, exact-gap graded-cap solves, and same-chain final
   symmetric truth run.

## Forensic lane

Task `97014` is the preferred old N1=6 forensic lane because it is the rank-1
geometry and was among the earliest starters. If it reaches terminal before
cutover, preserve its RESULT_JSON/stdout/stderr and classify it with the new
truth gate. If it is still non-terminal when gates 1--4 pass, the already
captured local exact-standard GUI interface/wall/5000 K evidence substitutes
for an old terminal, so `97014` is cancelled with the rest instead of holding a
compute slot.

## Resource outcome

The submit-before-cancel order keeps work continuously present in Scheduler.
Cancelling the 25 invalid old lanes after the 24 corrected replacements are
accepted releases up to 200 CPUs and about 1.6 TiB of requested memory for the
corrected campaign while retaining maximum campaign-level parallelism.
