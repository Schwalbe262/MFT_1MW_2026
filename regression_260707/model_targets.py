
"""Canonical surrogate-model target names.

The thermal extractor keeps the historical ``Tprobe_core_center_max`` column
as the hottest value across all core probe regions.  The three physical core
regions are also independent model targets so optimization and reporting do
not hide a local hot spot behind that aggregate.
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
