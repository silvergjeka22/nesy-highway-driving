"""Lab 5 — Control Barrier Function safety filter on the velocity command.

The continuous twin of the discrete safety shield: given the policy's desired
``(v, ω)``, clamp it to the nearest command that keeps a safety function
``h(x) ≥ 0``. The barrier uses the SAME RSS safe-distance formula as the discrete
``too_close`` predicate, so the CBF and the shield encode ONE rule (RG1) in two
representations — the project's "one rule, three encodings" demonstration.

Function-only. The 1-D longitudinal reduction of the CBF condition
``ḣ(x,u) ≥ -γ h(x)`` gives a closed-form speed bound, which is what a full QP
``min ||u - u_des||²`` collapses to when the leader is the only active constraint.
"""


def barrier_h(scene, cfg):
    """RG1 safety value ``h(x)`` = (gap to leader) − RSS safe-distance. ``h ≥ 0`` is safe.

    Uses ``nesy.roadmap.safe_distance`` — the SAME formula behind the discrete
    ``too_close`` predicate — so ``h < 0`` iff the shield's ``too_close`` is True.
    Returns +inf when there is no leader (unconstrained).
    """
    from nesy.roadmap import nearest_leader, safe_distance

    ego = scene["ego"]
    leader = nearest_leader(ego, scene.get("others", []))
    if leader is None:
        return float("inf")
    gap = (leader["x"] - ego["x"]) - cfg["rules"]["car_length"]
    return gap - safe_distance(ego["v"], leader["v"], cfg["rules"])


def cbf_filter(v_cmd, omega_cmd, scene, cfg):
    """Filter a desired ``(v, ω)`` to a command that respects the hard rules.

    Enforces, on the continuous action, the same hard constraints as the discrete
    shield (thresholds in ``cfg['cbf']`` are robot-scale, matching MetaDrive's
    ``(v, ω)``; ``safe_distance`` is scale-free SI so RG1 is shared verbatim):
      * **RG3** — never command above the legal speed limit ``cbf.v_max``.
      * **RG1** — CBF: with ``h = gap − safe_distance`` and ``ḣ ≈ v_lead − v``,
        ``ḣ ≥ -γ h`` gives the admissible speed bound ``v ≤ v_lead + γ h``.
      * **stay-on-road** — if off road, force a slow corrective speed.
      * **RI1** — do not gratuitously crawl to a stop: keep ``v ≥ cbf.v_min``,
        unless a hard rule above already required braking (safety wins).

    Steering ``ω`` is only clamped to the actuator limit here; the lateral gap is
    handled by the velocity-obstacle veto in ``nesy.roadmap.continuous_shield``.

    Returns ``((v_safe, ω_safe), intervened: bool)``.
    """
    from nesy.roadmap import nearest_leader

    c = cfg["cbf"]
    ego = scene["ego"]
    intervened = False

    omega_safe = float(max(-c["omega_max"], min(c["omega_max"], omega_cmd)))

    # RG3 — legal speed limit (robot scale).
    v_safe = float(v_cmd)
    if v_safe > c["v_max"]:
        v_safe = float(c["v_max"])
        intervened = True

    # RG1 — safe-distance barrier (only active with a leader).
    h = barrier_h(scene, cfg)
    if h != float("inf"):
        leader = nearest_leader(ego, scene.get("others", []))
        v_lead = leader["v"] if leader is not None else 0.0
        v_bound = v_lead + c["gamma"] * h
        if v_safe > v_bound:
            v_safe = float(max(0.0, v_bound))
            intervened = True

    # stay-on-road — corrective slow-down.
    if not ego.get("on_road", True):
        v_safe = min(v_safe, 0.5 * c["v_max"])
        intervened = True

    # RI1 — no gratuitous standstill (only when safety did not already demand braking).
    v_min = c.get("v_min", 0.0)
    if not intervened and v_safe < v_min:
        v_safe = float(v_min)
        intervened = True

    return (v_safe, omega_safe), intervened
