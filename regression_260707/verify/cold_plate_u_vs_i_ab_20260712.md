# Cold-plate U vs I A/B evidence — 2026-07-12

## Scope and identity

- U baseline: scheduler task `26607`, solver `e5b8f308c5a2507e2af92ffd01aec0a13a7572f0`
- Final side+center I: scheduler task `27399`, solver `3216e43a5a1a362ee2ed1aba89b642498c60d1b9`
- Shared PyAEDT library: `e6b9b9d20a832ff5c3f7ca97218737a0b8650781`
- Intermediate task `26614` is intentionally excluded because it used the discarded outer+outer I topology.
- Transformer dimensions, turns, gaps, material volumes/masses, electrical loading, air/cold-plate temperatures, and fan velocity match. The physical cold-plate stack is Al 20 T plus independent 2 T thermal pads; the old U solver represented this with legacy total-thickness semantics (`24`), while the final I solver uses aluminum thickness `20` and pads `2` separately.
- Final I task `27399` reached scheduler state `completed` with exit code `0` at 2026-07-12 05:39:23 KST.

Both rows were fetched with `regression_260707/verify/scheduler_client.py` using the expected solver and library revisions. `fetch_result(...).state == "valid"` and `is_valid_result(...) == True` for both. EM and thermal validity are `1`, required thermal values missing are `0`, and matrix/loss/thermal attempts are `1/1/1`.

## Results

For the eighth-symmetry model, the reported `Llt` and `Lmt` were multiplied by 2 to obtain the physical inductances.

| Quantity | U task 26607 | I task 27399 | I − U |
|---|---:|---:|---:|
| Physical leakage inductance | 63.350836 uH | 63.355557 uH | +0.004721 uH (+0.00745%) |
| Physical magnetizing inductance | 7681.335137 uH | 7687.687424 uH | +6.352286 uH (+0.08270%) |
| Coupling coefficient | 0.995901647 | 0.995904709 | +0.000003062 (+0.00031%) |
| Mean core flux density | 0.452403 T | 0.452754 T | +0.0775% |
| Maximum core flux density | 1.822294 T | 1.863932 T | +2.2850% |
| Winding loss | 3793.896 W | 3727.078 W | -66.818 W (-1.761%) |
| Core loss | 1437.804 W | 1441.594 W | +3.790 W (+0.264%) |
| Core cold-plate loss | 1677.649 W | 824.720 W | -852.929 W (-50.841%) |
| Winding cold-plate loss | 163.511 W | 150.025 W | -13.486 W (-8.248%) |
| Sum of the four loss components | 7072.861 W | 6143.417 W | -929.443 W (-13.141%) |
| Tx maximum temperature | 95.062 C | 95.942 C | +0.880 C |
| Rx-main maximum temperature | 68.365 C | 66.523 C | -1.841 C |
| Rx-side maximum temperature | 113.178 C | 111.104 C | -2.074 C |
| Core maximum temperature | 75.543 C | 100.930 C | +25.387 C |

The I-topology core probes were 85.256 C at the center leg, 85.316 C at the side leg, and 100.352 C at the top yoke. The thermal penalty is therefore concentrated at the top yoke, whose U-shaped bridge cooling path was removed.

Convergence was comparable: matrix `10/1` passes/consecutive, loss `4/2`, thermal `134` iterations, all residuals within contract, and mesh counts within 0.66%. Runtime was 3720.97 s for U and 6303.31 s for I, but the I run shared a saturated node (`254/256` CPUs in use and up to `64/64` FEA CPUs requested), so runtime difference is not attributed to geometry.

## Conclusion

The final I topology is electromagnetically neutral at this sample and reduces cold-plate eddy loss enough to lower the summed loss by about 13.1%. It is not thermally equivalent: core maximum temperature rises by 25.4 C because the top yoke becomes the hot spot. Retaining the I topology therefore needs an additional top-yoke cooling path or a thermal-interface redesign.

These diagnostic A/B tasks must not be ingested into training data.
