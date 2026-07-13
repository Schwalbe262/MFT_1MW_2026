"""Canonical surrogate-model target names.

The thermal extractor keeps the historical ``Tprobe_core_center_max`` column
as the hottest value across all core probe regions.  The three physical core
regions are also independent model targets so optimization and reporting do
not hide a local hot spot behind that aggregate.

For the versioned Rx-side face-probe contract,
``Tprobe_Rx_side_leeward_max`` is the hottest max across every physical side
winding's transformer-outward and transformer-inward radial probe.  Its paired
mean comes from that same selected face; unlike face areas are never averaged.
Older outward-only rows lack the probe-contract provenance and are rejected by
the current strict quality gate while remaining intact in the raw dataset.
"""

from __future__ import annotations


CORE_REGION_TEMPERATURE_TARGETS = (
    "Tprobe_core_center_leg_max",
    "Tprobe_core_side_leg_max",
    "Tprobe_core_top_yoke_max",
)

SURROGATE_TEMPERATURE_TARGETS = (
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_Rx_side_leeward_max",
    "Tprobe_core_center_max",
    *CORE_REGION_TEMPERATURE_TARGETS,
)

MANDATORY_SURROGATE_TEMPERATURE_TARGETS = (
    "Tprobe_Tx_leeward_max",
    "Tprobe_Rx_main_leeward_max",
    "Tprobe_core_center_max",
    *CORE_REGION_TEMPERATURE_TARGETS,
)


SURROGATE_TARGET_SEMANTICS = {
    "Tprobe_Rx_side_leeward_max": (
        "max_across_all_transformer_inner_and_outer_faces"
    ),
}


# These are direct, full-physical loss outputs emitted by the loss extractor.
# They must be trained independently; deriving a primary/secondary split from
# the aggregate winding-loss surrogate would invent information that the FEA
# did not provide.
SURROGATE_WINDING_COMPONENT_LOSS_TARGETS = (
    "P_Tx_main_group",
    "P_Rx_main_group",
    "P_Rx_side_total",
)
