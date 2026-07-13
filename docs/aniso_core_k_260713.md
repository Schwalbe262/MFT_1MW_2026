# Anisotropic thermal conductivity for the wound amorphous core

## Physics basis

The 2605SA1 core is a wound ribbon core. The ribbon width spans global `y`
throughout the core, while successive ribbon layers stack radially in the
`xz` magnetic-loop plane. Thermal transport is therefore high in the two
ribbon-plane directions and low normal to the layer stack.

The existing five-piece core topology supplies the required local orientation
without rotating material coordinate systems:

| Core piece | Lamination normal | `(kx, ky, kz)` |
| --- | --- | --- |
| `core_<g>_leg_left`, `leg_center`, `leg_right` | global `x` | `(k_throughstack, k_inplane, k_inplane)` |
| `core_<g>_yoke_top`, `yoke_bottom` | global `z` | `(k_inplane, k_inplane, k_throughstack)` |

All boxes use their default global-aligned orientation. This mapping matches
the electromagnetic segmentation, whose leg and yoke stacking directions are
respectively `V(1)` and `V(3)`.

## Effective-property model

For lamination factor `kf`, alloy conductivity `k_alloy`, and interlayer
conductivity `k_interlayer`, the model derives the properties at runtime:

```text
k_inplane = kf*k_alloy + (1-kf)*k_interlayer

k_throughstack = 1 / (kf/k_alloy + (1-kf)/k_interlayer)
```

The parallel rule of mixtures applies along the ribbon direction and ribbon
width. The series (harmonic) rule applies through the ribbon/interlayer stack.
No derived conductivity is hardcoded.

Default anchors are:

- `core_lamination_factor = 0.85`, the existing core material input.
- `core_k_alloy = 9.0 W/mK`. Metglas reports approximately `9 W/m-C` for
  2605SA1 over 20–100 °C in its [manufacturer FAQ](https://metglas.com/frequently-asked-questions/);
  the alloy and ribbon physical-property context is recorded in the
  [2605SA1 technical bulletin](https://metglas.com/wp-content/uploads/2021/06/2605SA1-Magnetic-Alloy-Updated.pdf).
- `core_k_interlayer = 0.2 W/mK`, an engineering anchor for epoxy/varnish that
  deliberately follows the existing `thermal_pad` material convention. It is
  not claimed as a Metglas alloy property.

These defaults produce:

```text
k_inplane     = 7.68 W/mK
k_throughstack = 1.184210526... W/mK
```

The rounded values are documentation only; Icepak receives the values derived
from each row's inputs.

## Solver implementation and legacy path

`core_k_anisotropic = 1` creates two Icepak materials and uses the segmented
core geometry:

```python
leg_material.thermal_conductivity = [
    k_throughstack, k_inplane, k_inplane,
]
yoke_material.thermal_conductivity = [
    k_inplane, k_inplane, k_throughstack,
]
```

A three-element Python list is PyAEDT's supported Cartesian anisotropic
`AnisoProperty` form (`component1`, `component2`, `component3` = global
`x`, `y`, `z`). Density and specific heat retain the prior values.

Maxwell still supplies one `P_core_<g>` loss per depth group. Icepak distributes
that retained group total across its live leg/yoke pieces by post-symmetry
volume. This preserves both the group total and uniform volumetric source
density.

`core_k_anisotropic = 0` retains the former unsegmented `core_<g>` geometry and
scalar `core_amorphous_thermal` material using `core_k_thermal`. That is the
explicit compatibility path for reproducing legacy thermal behavior.

## Result provenance and cohort rule

Every normal or non-converged thermal payload records:

- `thermal_core_conductivity_model`, either `isotropic_legacy` or
  `anisotropic_wound_rule_of_mixtures_v1`;
- `thermal_core_k_inplane` in W/mK;
- `thermal_core_k_throughstack` in W/mK.

These fields are also retained in the fixed training-I/O projection. Thermal
rows produced before and after this change must **not** be mixed when training
or evaluating `Tprobe_*` targets. Select a single
`thermal_core_conductivity_model` cohort (and retain the two derived values as
an audit check) before forming a temperature dataset. The model tag separates
the physical thermal policy even though the sealed geometry candidate digest
is intentionally unchanged.

## Verification boundary

The installed PyAEDT 0.22 implementation and its Icepak gRPC system tests accept
the same three-list anisotropic property form already used for homogenized
windings. No gRPC-specific workaround is indicated. This change has not run a
live AEDT 2026 R1 Icepak solve in this worktree, so native axis-effect readback
and thermal contact across the touching segmented solids remain the final
solver-level confirmation points.
