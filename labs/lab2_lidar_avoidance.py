"""Lab 2 — reactive LIDAR avoidance + range-to-predicate helpers.

The reactive last-resort safety floor, and the grounding of ``too_close`` /
``off_road`` from raw LIDAR ranges (MetaDrive exposes a lidar observation).

Function-only. The exact Lab 2 lidar message layout (sector ordering, max range)
was not available, so the parsing assumes a simple 1-D array of ranges evenly
spaced over 360°, with a TODO to align to the real handout API.
"""

import numpy as np


def lidar_to_predicates(ranges, cfg, d_too_close=None):
    """Derive reactive predicates from a 1-D LIDAR range array.

    Args:
        ranges: array of distances [m], evenly spaced around the robot.
        cfg: full config.
        d_too_close: override for the "too close" distance; defaults to
            ``cfg['cbf']['d_safe_min']``.

    Returns:
        dict with ``too_close`` (any range below threshold) and ``min_range``.
    """
    ranges = np.asarray(ranges, dtype=float)
    ranges = ranges[np.isfinite(ranges)]
    d = d_too_close if d_too_close is not None else cfg.get("cbf", {}).get("d_safe_min", 5.0)
    min_range = float(ranges.min()) if ranges.size else float("inf")
    return {"too_close": min_range < d, "min_range": min_range}


def reactive_avoidance(ranges, cfg):
    """Last-resort ``(v, ω)`` that steers away from the nearest obstacle.

    Steers toward the most-open sector and slows proportionally to how close the
    nearest return is. This only fires as the floor beneath the CBF/VO layers.

    Returns ``(v, ω)``.
    """
    md = cfg.get("metadrive", {})
    v_max = md.get("v_max", 10.0)
    omega_max = md.get("omega_max", 0.6)

    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0:
        return (v_max, 0.0)
    n = ranges.size
    nearest_i = int(np.nanargmin(ranges))
    openest_i = int(np.nanargmax(ranges))

    # Map sector index to a steering sign: positive angle steers left.
    angle = (openest_i - n / 2) / (n / 2)  # in [-1, 1]
    omega = float(np.clip(angle, -1.0, 1.0) * omega_max)

    # Slow down the closer the nearest obstacle is.
    near = float(ranges[nearest_i])
    d_safe = cfg.get("cbf", {}).get("d_safe_min", 5.0)
    v = float(v_max * np.clip(near / (2 * d_safe), 0.0, 1.0))

    # TODO: align sector->angle mapping and units to the actual Lab 2 lidar API.
    return (v, omega)
