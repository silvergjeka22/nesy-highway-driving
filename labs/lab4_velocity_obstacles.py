"""Lab 4 — velocity obstacles (VO): the continuous "is this gap safe?" test.

Grounds the lane-change / turn safety of the continuous ``(v, ω)`` command from
kinematics (Part 3). ``continuous_shield`` (nesy.roadmap) uses ``gap_is_safe`` to
veto a turn into a neighbour on a collision course — the continuous analogue of
the discrete shield gating ``LANE_LEFT``/``LANE_RIGHT`` on ``safe_gap``.

Function-only. Standard collision-cone maths; no top-level execution.
"""

import numpy as np


def in_velocity_obstacle(p_rel, v_rel, radius, horizon):
    """True iff the obstacle's relative motion leads to a collision within ``horizon``.

    Collision-cone test in the ego frame. ``p_rel`` is the obstacle position
    relative to the ego and ``v_rel = v_obstacle − v_ego`` its relative velocity,
    so its future position is ``p_rel + v_rel · t``. Closest approach within the
    combined ``radius`` and within ``horizon`` seconds means a collision.

    Args:
        p_rel: obstacle position relative to ego, (dx, dy) [m].
        v_rel: obstacle velocity relative to ego, (dvx, dvy) [m/s].
        radius: combined collision radius [m].
        horizon: look-ahead time [s].
    """
    p = np.asarray(p_rel, dtype=float)
    v = np.asarray(v_rel, dtype=float)
    vv = float(v @ v)
    if vv < 1e-9:
        return float(p @ p) <= radius ** 2  # not closing; unsafe only if overlapping
    t_star = float(np.clip(-(p @ v) / vv, 0.0, horizon))  # time of closest approach
    closest = p + v * t_star
    return float(closest @ closest) <= radius ** 2


def gap_is_safe(ego, neighbours, cfg, horizon=None, radius=None):
    """True iff no vehicle in ``neighbours`` is on a collision course with the ego (VO).

    Continuous replacement for the kinematic ``nesy.roadmap.safe_gap`` test. ``ego``
    and each neighbour are dicts with ``x, y, vx, vy``.
    """
    vo = cfg.get("vo", {})
    horizon = horizon if horizon is not None else vo.get("time_horizon", 3.0)
    radius = radius if radius is not None else vo.get("radius", 1.5)
    for o in neighbours:
        p_rel = (o["x"] - ego["x"], o.get("y", 0.0) - ego.get("y", 0.0))
        v_rel = (o["vx"] - ego["vx"], o.get("vy", 0.0) - ego.get("vy", 0.0))
        if in_velocity_obstacle(p_rel, v_rel, radius, horizon):
            return False
    return True
