"""Lab 4 — velocity obstacles (VO/RVO) + optional MCTS manoeuvre search.

The continuous, geometric version of "is this gap safe?". Grounds
``safe_gap`` from continuous kinematics (Part 3) and acts as a fallback
collision-avoidance layer when the CBF model is too coarse.

Function-only. The VO maths is standard; the MCTS planner is left as a clearly
marked stub (the Lab 4 handout API was not available).
"""

import numpy as np


def in_velocity_obstacle(p_rel, v_rel, radius, horizon):
    """True iff the obstacle's relative motion leads to a collision within ``horizon``.

    Collision cone test in the ego frame. ``p_rel`` and ``v_rel`` must be in the
    SAME (obstacle-relative-to-ego) convention: the obstacle position is
    ``p_rel`` and it drifts at ``v_rel`` per second, so its future position is
    ``p_rel + v_rel · t``. Closest approach within the combined ``radius`` and
    within ``horizon`` means a collision.

    Args:
        p_rel: obstacle position relative to ego, (dx, dy) [m].
        v_rel: obstacle velocity relative to ego (v_obstacle - v_ego), (dvx, dvy).
        radius: combined collision radius [m].
        horizon: look-ahead time [s].
    """
    p = np.asarray(p_rel, dtype=float)
    v = np.asarray(v_rel, dtype=float)
    vv = float(v @ v)
    if vv < 1e-9:
        return float(p @ p) <= radius ** 2  # not closing; only unsafe if overlapping
    # time of closest approach (clamped to [0, horizon]); obstacle pos = p + v t
    t_star = float(np.clip(-(p @ v) / vv, 0.0, horizon))
    closest = p + v * t_star
    return float(closest @ closest) <= radius ** 2


def gap_is_safe(ego, lane_vehicles, cfg, horizon=None, radius=None):
    """True iff no vehicle in ``lane_vehicles`` is on a collision course (VO).

    Continuous replacement for the kinematic ``nesy.roadmap.safe_gap`` test, used
    for lane-change decisions in MetaDrive. ``ego`` and each vehicle are dicts
    with ``x, y, vx, vy``.
    """
    vo = cfg.get("vo", {})
    horizon = horizon if horizon is not None else vo.get("time_horizon", 3.0)
    radius = radius if radius is not None else vo.get("radius", 1.5)
    for o in lane_vehicles:
        p_rel = (o["x"] - ego["x"], o.get("y", 0.0) - ego.get("y", 0.0))
        v_rel = (o["vx"] - ego["vx"], o.get("vy", 0.0) - ego.get("vy", 0.0))
        if in_velocity_obstacle(p_rel, v_rel, radius, horizon):
            return False
    return True


def vo_admissible(v_cmd, ego, neighbours, cfg):
    """Project a desired ``(vx, vy)`` out of any active velocity obstacle.

    Minimal fallback avoidance: if the commanded velocity is inside a VO, scale
    it down (brake) until it is admissible. A full RVO solver would instead pick
    the nearest admissible velocity on the VO boundary.
    """
    vo = cfg.get("vo", {})
    horizon = vo.get("time_horizon", 3.0)
    radius = vo.get("radius", 1.5)
    v = np.asarray(v_cmd, dtype=float)
    for _ in range(10):
        unsafe = any(
            in_velocity_obstacle(
                (o["x"] - ego["x"], o.get("y", 0.0) - ego.get("y", 0.0)),
                (o["vx"] - v[0], o.get("vy", 0.0) - v[1]),
                radius,
                horizon,
            )
            for o in neighbours
        )
        if not unsafe:
            return tuple(v)
        v *= 0.7  # brake toward zero until admissible
    return (0.0, 0.0)


def mcts_plan(scene, cfg, depth=3):
    """Optional symbolic planner over manoeuvre sequences (Lab 4 MCTS).

    TODO: implement Monte-Carlo Tree Search over the discrete manoeuvre set,
    scoring rollouts by progress while pruning any node the shield rejects. The
    Lab 4 handout API was unavailable, so this is intentionally a stub.
    """
    raise NotImplementedError("TODO: Lab 4 MCTS manoeuvre search (handout API unknown).")
