"""Lab 5 — Control Barrier Function safety filter on the velocity command.

The continuous twin of the discrete safety shield: given the policy's desired
``(v, ω)``, project it to the nearest command that keeps a safety function
``h(x) ≥ 0`` (safe distance to the leader; on-road). Showing this CBF and the
discrete shield agree on the same scenario is the project's "one rule, three
encodings" demonstration.

Function-only. A full implementation solves a small QP
``min ||u - u_des||²  s.t.  ḣ(x,u) ≥ -γ h(x)``; here we use a closed-form
longitudinal projection (decelerate when the safe-distance barrier is active),
which is the common 1-D reduction. The full multi-constraint QP is marked TODO.
"""


def barrier_h(scene, cfg):
    """Safety value ``h(x)`` = (gap to leader) − d_safe_min. ``h >= 0`` is safe.

    Returns +inf when there is no leader (unconstrained).
    """
    c = cfg["cbf"]
    ego = scene["ego"]
    others = scene.get("others", [])
    ahead = [o for o in others if o.get("lane") == ego.get("lane") and o["x"] > ego["x"]]
    if not ahead:
        return float("inf")
    leader = min(ahead, key=lambda o: o["x"] - ego["x"])
    gap = (leader["x"] - ego["x"]) - cfg["rules"]["car_length"]
    return gap - c["d_safe_min"]


def cbf_filter(v_cmd, omega_cmd, scene, cfg):
    """Filter a desired ``(v, ω)`` to a safe command.

    Longitudinal CBF: with ``h = gap - d_safe_min`` and the leader treated as the
    constraint, enforce ``ḣ ≥ -γ h``. Approximating ``ḣ ≈ v_leader - v`` gives an
    upper bound on the admissible speed; we clamp ``v`` to it. Steering ``ω`` is
    passed through (lateral safety is handled by the VO layer / on-road check).

    Returns ``((v_safe, ω_safe), intervened: bool)``.
    """
    c = cfg["cbf"]
    ego = scene["ego"]
    h = barrier_h(scene, cfg)

    v_safe = float(min(v_cmd, c["v_max"]))
    omega_safe = float(max(-c["omega_max"], min(c["omega_max"], omega_cmd)))

    intervened = False
    if h != float("inf"):
        others = scene.get("others", [])
        ahead = [o for o in others if o.get("lane") == ego.get("lane") and o["x"] > ego["x"]]
        leader = min(ahead, key=lambda o: o["x"] - ego["x"]) if ahead else None
        v_lead = leader["v"] if leader is not None else 0.0
        # Admissible speed bound from h_dot >= -gamma * h  (h_dot ~ v_lead - v).
        v_bound = v_lead + c["gamma"] * h
        if v_safe > v_bound:
            v_safe = float(max(0.0, v_bound))
            intervened = True

    # off-road barrier: if off road, force a slow, corrective command.
    if not ego.get("on_road", True):
        v_safe = min(v_safe, 0.5 * c["v_max"])
        intervened = True

    # TODO: replace the 1-D longitudinal projection with the full QP
    #   min ||u - u_des||^2  s.t.  L_f h + L_g h u >= -gamma * h   (per obstacle),
    # using the Lab 5 barrier/dynamics definitions once the handout API is known.
    return (v_safe, omega_safe), intervened
