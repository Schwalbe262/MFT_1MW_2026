"""Versioned semantics for the Rx-side winding thermal face probes.

The side winding exists at negative x in symmetry models and at both negative
and positive x in a full model.  ``outward`` always means away from the
transformer centre (x=0), while ``inward`` means toward it.  The representative
temperature is deliberately selected from the hottest face; face means are
never averaged because a future geometry may give the faces different areas.
"""

from __future__ import annotations

import math


RX_SIDE_FACE_PROBE_CONTRACT_VERSION = "rx-side-transformer-inner-outer-v1"
RX_SIDE_FACE_MAX_RULE = "max_across_all_transformer_inner_and_outer_faces"
RX_SIDE_FACE_MEAN_RULE = "mean_of_face_selected_by_max_no_cross_face_average"


def select_hottest_complete_face(temperatures, face_names):
    """Return ``(maximum, paired_mean, name)`` only for a complete face set.

    Requiring both max and mean for every requested face prevents a missing
    inward face from silently falling back to the historical outward-only
    result.  The returned mean belongs to the exact face that supplied the
    maximum; it is not an unweighted average across potentially unequal areas.
    """
    names = tuple(dict.fromkeys(str(name) for name in face_names))
    if not names:
        return None
    candidates = []
    for name in names:
        try:
            maximum = float(temperatures[f"{name}_max"])
            mean = float(temperatures[f"{name}_mean"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not (math.isfinite(maximum) and math.isfinite(mean)):
            return None
        candidates.append((maximum, mean, name))
    return max(candidates, key=lambda item: item[0])


def aggregate_rx_side_faces(temperatures, outward_names, inward_names):
    """Return complete aggregate fields plus the representative face name.

    An empty mapping and empty name are returned until both transformer-
    relative face sets are complete.  This makes partial extraction unusable
    as a deceptively low temperature target.
    """
    outward_names = tuple(outward_names)
    inward_names = tuple(inward_names)
    if not outward_names or len(outward_names) != len(inward_names):
        return {}, ""
    outward = select_hottest_complete_face(temperatures, outward_names)
    inward = select_hottest_complete_face(temperatures, inward_names)
    representative = select_hottest_complete_face(
        temperatures, outward_names + inward_names
    )
    if outward is None or inward is None or representative is None:
        return {}, ""
    return {
        "Tprobe_Rx_side_outer_max": outward[0],
        "Tprobe_Rx_side_outer_mean": outward[1],
        "Tprobe_Rx_side_inner_max": inward[0],
        "Tprobe_Rx_side_inner_mean": inward[1],
        "Tprobe_Rx_side_leeward_max": representative[0],
        "Tprobe_Rx_side_leeward_mean": representative[1],
    }, representative[2]
