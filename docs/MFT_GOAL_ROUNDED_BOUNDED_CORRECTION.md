# Rounded candidate #5 bounded correction (prepare only)

`tools/mft_goal_official5_rounded_bounded_correction_prepare.py` is a
fail-closed contingency planner for one case only: authenticated rounded
candidate #5 task `96340` completes successfully, retains `symmetric.aedt`,
but narrowly misses one or more requested design constraints.

It does not submit, retry, cancel, or mutate a Scheduler task. The CLI exposes
only `prepare`, the emitted manifests record zero Scheduler methods and zero
POST calls, and no automatic continuation is present. A separate reviewed
submission step would be required to run any prepared neighbor.

## Admission gate

All of the following must be true before a plan is written:

- exact task `96340`, candidate physics SHA
  `909d249ebe455d6f60b42d094e7916c8b3e8538e8d188e48a3906d82665ebc42`,
  reviewed solver revision, node, dedupe key, terminal success, retained
  `symmetric.aedt`, result parameters, and collection lineage authenticate;
- rounded eighth-symmetry identity remains `R10/s4`;
- cooling remains dual fan at `1.5 m/s`, `k_ins=0.2 W/mK`, and both pads at
  `2.0 mm`;
- all exterior dimensions already pass `1200 x 1000 x 750 mm`;
- at least one requested resonance or temperature constraint misses;
- resonance is at least `14900 Hz`, winding maximum is at most `100.5 C`,
  and core maximum is at most `120.5 C`;
- surrogate-to-FEA residuals are no larger than `20 L`, `300 W`, `500 Hz`,
  `3 C` winding, and `3 C` core;
- a simultaneous multi-constraint miss is additionally limited to `75 Hz`,
  `0.25 C` winding, and `0.20 C` core deficit/excess.

A passing result, a dimension failure, an unauthenticated artifact, a broad
model mismatch, or a non-finite value stops without producing a plan.

## Exact neighborhood

The selected chromosome is replayed as a fixed point through
`Current7Tier1Problem.repair_unit_coordinates` and
`decode_unit_sample_with_cw1`. Turns and topology remain fixed at
`N1=6`, `N2=37+23`, and five core groups. Rounding, cooling, and symmetry
cannot change.

Only four deterministic same-design templates are considered:

| Template | Repaired intent | Principal decoded change |
| --- | --- | --- |
| A | balanced resonance and temperature | `cw1 1.13 -> 1.00`, `w1 476 -> 506 mm` |
| B | thermal margin | `w1 476 -> 506 mm` |
| C | resonance margin | `cw1 1.13 -> 1.00`, `nwh2 285.6 -> 290.1 mm` |
| D | core temperature margin | `l1 68 -> 69 mm`, `w1 476 -> 488 mm` |

Every chromosome remains within an `L-infinity` radius of `0.05`. The
authenticated terminal population supplies an anchor-centred weighted ridge
delta model. The measured task `96340` residual is decayed with distance and
applied to each local prediction.

The older `tools/mft_goal_local_trust_acquisition.py` neighbor path is
explicitly rejected. Its raw decoder replays the sealed anchor as
`cw1=4.55`, `cw2=0.758`, and `wcp_len_x=332.0 mm`, rather than
`1.13`, `1.10`, and `349.6 mm`. It must not be used for this candidate.

## Selection and stopping

A candidate survives only if its corrected prediction meets all conservative
guards:

- resonance `>=15020 Hz`;
- winding maximum `<=99.8 C`;
- core maximum `<=119.8 C`;
- dimensions `<=1199 x 999 x 749 mm`.

The tool ranks surviving, positively improving templates and emits at most
three candidates. It authorizes one correction round, parallel symmetric
validation only. It forbids surrogate retraining, full-model validation,
turn/core-group changes, cooling changes, and rounding changes.

## Invocation

```powershell
python tools/mft_goal_official5_rounded_bounded_correction_prepare.py prepare `
  --standard-plan <task96340-standard-plan.json> `
  --standard-final <task96340-submission-final.json> `
  --standard-collection <authenticated-task96340-collection-directory> `
  --output <new-empty-output-directory>
```

The output directory must not already exist. The immutable package contains:

- `authenticated_task96340_source.json`;
- `narrow_miss_gate.json`;
- `legacy_local_trust_path_rejection.json`;
- up to three ranked `candidate_<rank>_<template>.json` files;
- `bounded_correction_prepare_plan.json`;
- `prepare_receipt.json`.

Each JSON object is SHA-256 sealed. The plan is evidence for review, not
Scheduler submission authority.
