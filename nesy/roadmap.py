"""NeSy core: predicates, safety shield, independent monitor, penalty.

Operates on a scene dict in SI units (metres, m/s). Rule parameters from
cfg['rules'] (Maierhofer et al. 2020, Table II).
"""

ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTION_INDEX = {v: k for k, v in ACTIONS.items()}


def predicates(scene, cfg):
    """Ground a scene into truth-valued predicates."""
    r = cfg["rules"]
    ego = scene["ego"]
    others = scene.get("others", [])

    leader = nearest_leader(ego, others)
    keeps_sd = keeps_safe_distance(ego, leader, r) if leader is not None else True

    preds = {
        # RG1
        "keeps_safe_distance": keeps_sd,
        "too_close": (leader is not None) and (not keeps_sd),
        "safe_gap_left": safe_gap(ego, others, side=-1, cfg=cfg),
        "safe_gap_right": safe_gap(ego, others, side=+1, cfg=cfg),
        # stay on road
        "off_road": not ego.get("on_road", True),
        # RG3
        "over_speed_limit": ego["v"] > r["v_max"],
        "speed_below_min": ego["v"] < (r["v_max"] - r["delta_v_fl"]),
        # RI1 (highway-env never fully stops, so "standstill" = crawling below v_stall)
        "in_standstill": abs(ego["v"]) <= r.get("v_stall", r["v_err"]),
        # RG2 (needs ego acceleration; False if unavailable)
        "abrupt_braking": _abrupt_braking(ego, r),
        # RG4
        "impedes_flow": _impedes_flow(ego, leader, r),
        # RI2
        "passing_on_right": _passing_on_right(ego, others, r),
        # RI3 (needs heading; stubbed -> False when heading absent)
        "makes_uturn": _makes_uturn(ego, r),
    }
    preds["faster_than_left"] = preds["passing_on_right"]
    preds["must_not_stop"] = preds["in_standstill"]
    preds["leader_gap"] = (
        (leader["x"] - ego["x"] - r["car_length"]) if leader is not None else float("inf")
    )
    return preds


def nearest_leader(ego, others, max_dx=None):
    """Nearest vehicle ahead in the ego's path (or None).

    In the path = same lane label, or laterally overlapping now or within 1 s
    (projected with relative lateral velocity — catches cut-ins and straddlers,
    whose lane label only flips halfway across).
    """
    def in_path(o):
        if o.get("lane") == ego.get("lane"):
            return True
        dy = o.get("y", 0.0) - ego.get("y", 0.0)
        dy_1s = dy + (o.get("vy", 0.0) - ego.get("vy", 0.0)) * 1.0
        # A yawed car's nose/tail sticks out laterally beyond its centre.
        yaw_extent = 2.5 * min(1.0, abs(o.get("heading", 0.0) - ego.get("heading", 0.0)))
        return min(abs(dy), abs(dy_1s)) - yaw_extent < 2.0

    cand = [o for o in others if o["x"] > ego["x"] and in_path(o)]
    if max_dx is not None:
        cand = [o for o in cand if (o["x"] - ego["x"]) <= max_dx]
    return min(cand, key=lambda o: o["x"] - ego["x"]) if cand else None


def safe_distance(v_follow, v_lead, r):
    """RSS-style legal safe distance (Maierhofer 2020)."""
    a_f = abs(r["a_min_ego"])
    a_l = abs(r["a_min_other"])
    return v_follow * r["t_d"] + v_follow ** 2 / (2 * a_f) - v_lead ** 2 / (2 * a_l)


def keeps_safe_distance(ego, leader, r):
    """True iff the gap to leader meets safe_distance."""
    gap = (leader["x"] - ego["x"]) - r["car_length"]
    return gap >= safe_distance(ego["v"], leader["v"], r)


def safe_gap(ego, others, side, cfg):
    """True iff a lane change to ego.lane + side is safe (side: -1=left, +1=right)."""
    r = cfg["rules"]
    target = ego.get("lane", 0) + side
    if target < 0 or target >= scene_lanes(cfg, ego):
        return False  # no such lane
    lane_vs = [o for o in others if o.get("lane") == target]
    ahead = [o for o in lane_vs if o["x"] >= ego["x"]]
    behind = [o for o in lane_vs if o["x"] < ego["x"]]
    ok = True
    if ahead:
        ok = ok and keeps_safe_distance(ego, min(ahead, key=lambda o: o["x"] - ego["x"]), r)
    if behind:
        ok = ok and keeps_safe_distance(max(behind, key=lambda o: o["x"]), ego, r)
    return ok


def scene_lanes(cfg, ego):
    """Number of lanes: from the scene when known (MetaDrive), else env config."""
    if ego.get("lanes_count"):
        return int(ego["lanes_count"])
    return cfg.get("env", {}).get("config", {}).get("lanes_count", 4)


def _abrupt_braking(ego, r):
    """RG2: harsh deceleration below a_abrupt."""
    a = ego.get("a")
    return False if a is None else a < r["a_abrupt"]


def _impedes_flow(ego, leader, r):
    """RG4: ego slower than flow without a slow leader forcing it."""
    if leader is not None and (ego["v"] - leader["v"]) > 0:
        return False
    return ego["v"] < (r["v_max"] - r["delta_v_fl"])


def _passing_on_right(ego, others, r):
    """RI2: ego faster than a vehicle to its left, outside exceptions."""
    left_lane = ego.get("lane", 0) - 1
    for o in [o for o in others if o.get("lane") == left_lane]:
        if ego["v"] <= o["v"]:
            continue
        if o["v"] <= r["v_qv"] and (ego["v"] - o["v"]) <= r["slightly_higher_speed"]:
            continue
        return True
    return False


def _makes_uturn(ego, r):
    """RI3: heading exceeds U-turn threshold."""
    h = ego.get("heading")
    return False if h is None else abs(h) > r["delta_theta_uturn"]


def safety_shield(action, preds, fsm_state, cfg):
    """Replace an unsafe manoeuvre with the safest legal fallback. Returns (action_idx, fsm_state)."""
    from labs.lab3_fsm import fsm_transition, admissible_actions

    fsm_state = fsm_transition(fsm_state, preds, cfg)
    proposed = ACTIONS[int(action)]
    allowed = admissible_actions(fsm_state, preds, cfg)
    if proposed in allowed:
        return ACTION_INDEX[proposed], fsm_state
    return ACTION_INDEX[safest_fallback(preds, cfg, allowed, state=fsm_state)], fsm_state


def safest_fallback(preds, cfg, allowed=None, state=None):
    """Brake if unsafe; in OVERTAKE_LEFT execute the pass; else hold lane."""
    if preds.get("too_close") or preds.get("off_road"):
        return "SLOWER"
    if state == "OVERTAKE_LEFT" and allowed and "LANE_LEFT" in allowed:
        return "LANE_LEFT"  # the FSM committed to the pass: execute it
    if allowed and "IDLE" in allowed:
        return "IDLE"
    return "SLOWER"


def continuous_shield(v_cmd, omega_cmd, scene, cfg, lidar_ranges=None):
    """Filter (v, ω) through CBF (Lab 5) + VO (Lab 4) + LIDAR (Lab 2). Returns ((v, ω), intervened)."""
    from labs.lab5_cbf import cbf_filter
    from labs.lab4_velocity_obstacles import gap_is_safe

    (v_safe, omega_safe), intervened = cbf_filter(v_cmd, omega_cmd, scene, cfg)

    if abs(omega_safe) > 1e-6 and not gap_is_safe(scene["ego"], scene.get("others", []), cfg):
        omega_safe = 0.0
        intervened = True

    if lidar_ranges is not None:
        from labs.lab2_lidar_avoidance import lidar_to_predicates, reactive_avoidance
        # Brake only for obstacles in the frontal ±60° sector: a car alongside in
        # the next lane is not in the path, and braking for it blocks every pass.
        n = len(lidar_ranges)
        frontal = lidar_ranges[n // 3: 2 * n // 3 + 1]
        lpreds = lidar_to_predicates(frontal, cfg)
        if lpreds["too_close"]:
            v_react, _ = reactive_avoidance(frontal, cfg)
            if v_react < v_safe:
                v_safe = v_react
                intervened = True

    return (v_safe, omega_safe), intervened


def rule_encoding_agreement(scene, cfg):
    """RG1 checked via MTL predicate, discrete shield mask, and CBF barrier — must agree."""
    from labs.lab5_cbf import barrier_h
    from labs.lab3_fsm import admissible_actions

    preds = predicates(scene, cfg)
    logic = bool(preds["too_close"])
    discrete = "FASTER" not in admissible_actions("CRUISE", preds, cfg)
    continuous = barrier_h(scene, cfg) < 0.0
    return {
        "too_close (MTL)": logic,
        "shield vetoes FASTER": discrete,
        "CBF barrier active": continuous,
        "agree": logic == discrete == continuous,
    }


def logic_penalty(preds, cfg):
    """Weighted sum of soft-rule violations (RI2, RG4, RG2). Returns >= 0."""
    lam = cfg["rules"]["lambda"]
    pen = 0.0
    if preds.get("passing_on_right"):
        pen += lam["RI2"]
    if preds.get("impedes_flow"):
        pen += lam["RG4"]
    if preds.get("abrupt_braking"):
        pen += lam["RG2"]
    return pen


def intersection_predicates(scene, cfg):
    """Intersection rules (2022 paper) — stubbed pending MetaDrive intersection API."""
    return {
        "must_stop_at_line": False,
        "light_is_red": False,
        "has_right_of_way": True,
        "yields_left_turn": True,
    }


def rule_violations(preds, cfg):
    """Per-rule boolean violations for one timestep (independent monitor)."""
    return {
        "RG1": bool(preds.get("too_close")),
        "RG2": bool(preds.get("abrupt_braking")),
        "RG3": bool(preds.get("over_speed_limit")),
        "RG4": bool(preds.get("impedes_flow")),
        "RI1": bool(preds.get("must_not_stop")),
        "RI2": bool(preds.get("passing_on_right")),
    }
