# Optional electrostatic capacitance stage (2026-07-13)

## What is implemented

`run_simulation_260706.py` accepts three fixed-run controls that are echoed in
the result row but deliberately excluded from the sealed campaign `KEYS`
identity:

- `cap_on` (default `0`)
- `cap_max_passes` (default `10`)
- `cap_percent_error` (default `1.0`)

The sealed 71-key candidate schema, the pre-stage 75-key full schema from
`e30c070`, and the current 78-key full schema remain authenticated; candidate
digests continue to project onto only the sealed 71 keys.

With `cap_on=1`, the normal AC-magnetic matrix design is built and solved
first. A fresh Maxwell 3D `Electrostatic` design named `maxwell_cap` is then
inserted in the same AEDT project. Physical solids are copied from
`maxwell_matrix`; magnetic terminal/flux sheets, magnetic boundaries, the
magnetic Matrix parameter, magnetic solution data, and the smaller magnetic
eighth-model air region are not. The electrostatic design creates a fresh,
matched air region as described below.

The electrostatic conductor model is intentionally lumped:

- every Tx turn is the `CapTx` equipotential net;
- every Rx turn is the `CapRx` equipotential net;
- amorphous-core solids and aluminum cooling plates are held at 0 V;
- only the remote air-region faces are held at 0 V;
- the native `CapMatrix` contains the two signal excitations, while the 0-V
  boundaries are excluded from the matrix sources.

Maxwell's signed 2x2 capacitance matrix is exported through
`export_c_matrix`. The payload preserves the signed off-diagonal coefficient
and also emits its positive magnitude for the interwinding estimate:

- raw retained-geometry values: `C_tx_tx_raw_F`, `C_rx_rx_raw_F`,
  `C_tx_rx_signed_raw_F`, `C_tx_rx_raw_F`;
- restored full-transformer values: `C_tx_tx_F`, `C_rx_rx_F`,
  `C_tx_rx_signed_F`, `C_tx_rx_F`;
- first-order estimates: `f_res_tx_self_Hz`, `f_res_rx_self_Hz`, and
  `f_res_interwinding_Hz`.

The resonance formula is evaluated in Python as
`1/(2*pi*sqrt(L_H*C_F))`. Tx and Rx use matrix-stage `Ltx` and `Lrx` with the
corresponding diagonal Maxwell C coefficient. The interwinding estimate uses
primary-referred leakage `Llt` and `abs(C_tx_rx)`.

The Maxwell diagonal coefficients are measured with the other winding net,
the core, the plates, and the remote region boundary all at 0 V. Consequently
`f_res_tx_self_Hz` and `f_res_rx_self_Hz` are **grounded-other-net screening
poles**, not open-circuit transformer self-resonances. A floating-other-net
capacitance would require a matrix reduction, and a real winding self-resonance
requires a distributed voltage/turn network. Likewise,
`f_res_interwinding_Hz` is a topology-dependent first-order heuristic formed
from primary-referred leakage L and mutual partial C; it is not an eigenmode of
a specified common-mode/differential-mode circuit. These interpretation and
ground policies are recorded directly in the capacitance payload.

The geometry has no explicit enamel/tape/interwinding-insulation solids; only
the existing thermal-pad solids carry their assigned dielectric property.
Therefore the air/vacuum winding gaps can materially bias C. The grounded air
region is also a finite enclosure, so diagonal capacitance remains
enclosure-dependent. Use this stage to screen candidates and harmonics, then
build a detailed dielectric and winding-network model for any close margin.

## Three-plane symmetry convention

For the retained domain (`x<=0`, `y>=0`, `z>=0`), the three cut region faces
receive explicit even electrostatic symmetry (`n.D=0`). The other three remote
region faces are grounded. PyAEDT percentage padding is relative to the current
model span, which is halved by each symmetry cut. The eighth electrostatic
region therefore uses 200% padding on each retained remote side; after
mirroring, it matches the full design's 100%-of-full-span padding. This avoids
using the magnetic eighth region's smaller enclosure.

With that matched boundary and the same conductor voltage on every mirror,
electrostatic energy and every C coefficient scale with retained volume:

```text
C_full = 8 * C_raw_eighth
```

Therefore `cap_capacitance_restoration_factor` is `8` for an eighth model and
`1` for a full model. Both raw and restored matrix entries are emitted. The
magnetic matrix in this repository has a different established restoration
contract, so `cap_inductance_restoration_factor` is `2` for an eighth model and
`1` for a full model. Resonance estimates always combine restored full-basis L
with restored full-basis C.

## Timing fields

The stage extends the additive `mft-stage-timing-v1` instrumentation with:

- `stage_time_cap_model_s`: clean design creation, solid copy, boundaries,
  matrix, and setup;
- `stage_time_cap_solve_s` / `cap_solve_time_s` / `time_cap`: AEDT Analyze
  time;
- `stage_time_cap_extract_s` / `cap_extraction_time_s`: native export, parse,
  restoration, and resonance calculations;
- `stage_time_cap_analyze_overhead_s` and
  `stage_time_cap_analyze_total_s`;
- `stage_time_cap_total_s` / `cap_stage_added_time_s`: the complete optional
  stage, including convergence export, from design creation through payload.

`ab_process_wall_s` from the A/B launcher remains the authoritative end-to-end
comparison because it also includes project saves, persistence, and shutdown.

## Exact timing A/B

Keep the baseline `cand.json` unchanged. Save this exact variant-only overlay
as `cap_on_overlay.json`:

```json
{"cap_on":1}
```

Run both arms sequentially inside one cluster allocation (this command launches
local subprocesses only; it does not submit anything to the scheduler):

```bash
python regression_260707/verify/run_efficiency_ab.py \
  --params cand.json \
  --overlay cap_on_overlay.json \
  --arm both \
  --output-dir ab/electrostatic_cap_260713/design_001
```

On the supplied Windows development environment, the equivalent exact command
is:

```powershell
& 'C:/Users/peets/anaconda3/envs/pyaedt2026v1/python.exe' regression_260707/verify/run_efficiency_ab.py `
  --params cand.json `
  --overlay cap_on_overlay.json `
  --arm both `
  --output-dir ab/electrostatic_cap_260713/design_001
```

Compare the outputs with:

```bash
python regression_260707/verify/compare_efficiency_ab.py \
  ab/electrostatic_cap_260713/design_001/baseline_result.json \
  ab/electrostatic_cap_260713/design_001/variant_result.json \
  --json-output ab/electrostatic_cap_260713/design_001/comparison.json
```

Compute added wall time as
`variant.ab_process_wall_s - baseline.ab_process_wall_s`. The comparator's
`saved_seconds` uses the opposite sign (`baseline - variant`), so a correctly
added electrostatic stage normally appears as a negative saving. Diagnose the
delta against the variant's `cap_stage_added_time_s` and its model/solve/extract
components.

Before measurement, a reasonable planning estimate is **5-15 added minutes per
design** at the default ten passes. This is deliberately provisional. The
electrostatic solve has no 1 kHz skin/proximity-loss mesh physics and is expected
to be well below the historical 27.15-minute median loss Analyze time, but the
many thin Rx turns and file-system/project-copy overhead make a cluster A/B the
only approval evidence. Run several geometrically diverse candidates and
alternate arm order before adopting a queue-time allowance.
