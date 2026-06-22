"""NeSy core (Part 2): predicates, safety shield, independent monitor, penalty.

Function-only module. Everything operates on a **scene dict** in SI units, not on
the normalised network observation — so the predicates are physically meaningful
(metres, m/s) and unit-testable on hand-built scenes. The env factories provide
``read_scene(env)`` adapters that emit this schema, so the *same* predicate
library is reused unchanged across highway-env (Parts 1-2) and MetaDrive (Part 3).

Scene schema
------------
    scene = {
        "ego":    {"x","y","vx","vy","v","a"?, "lane"(int), "on_road"(bool), "heading"?},
        "others": [{"x","y","vx","vy","v","lane"(int)}, ...],   # absolute, SI units
        "lanes_count": int,
    }

Rule parameters come from ``cfg['rules']`` (see nesy/RULES.md, traceable to
Maierhofer et al. 2020, Table II). No top-level execution.
"""

# DiscreteMetaAction index <-> manoeuvre name (highway-env convention).
ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
ACTION_INDEX = {v: k for k, v in ACTIONS.items()}


# =============================================================================
# Stage A — predicates (perception -> logic)
# =============================================================================
def predicates(scene, cfg):
    """Ground a scene into truth-valued (and a few float) predicates.

    Pure function: ``scene -> dict[str, bool|float]``. See module docstring for
    the scene schema and nesy/RULES.md for the rule mapping.
    """
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
        # RI1
        "in_standstill": abs(ego["v"]) <= r["v_err"],
        # RG2 (needs ego acceleration; False if unavailable)
        "unnecessary_braking": _unnecessary_braking(ego, leader, r),
        # RG4
        "impedes_flow": _impedes_flow(ego, leader, r),
        # RI2
        "passing_on_right": _passing_on_right(ego, others, r),
        # RI3 (needs heading; stubbed -> False when heading absent)
        "makes_uturn": _makes_uturn(ego, r),
    }
    preds["faster_than_left"] = preds["passing_on_right"]
    # "must not stop" = standing where stopping is forbidden. On an interstate the
    # exception is congestion / standing leader; simplified here to "standing and
    # the road ahead is clear". TODO(Part 3): use lane-type / congestion map.
    preds["must_not_stop"] = preds["in_standstill"] and (leader is None)
    # distance to leader (float; useful for the FSM and for differentiable logic)
    preds["leader_gap"] = (
        (leader["x"] - ego["x"] - r["car_length"]) if leader is not None else float("inf")
    )
    return preds


# ---- predicate helpers ------------------------------------------------------
def nearest_leader(ego, others, max_dx=None):
    """Nearest vehicle ahead of the ego in the same lane (or None)."""
    cand = [o for o in others if o.get("lane") == ego.get("lane") and o["x"] > ego["x"]]
    if max_dx is not None:
        cand = [o for o in cand if (o["x"] - ego["x"]) <= max_dx]
    return min(cand, key=lambda o: o["x"] - ego["x"]) if cand else None


def safe_distance(v_follow, v_lead, r):
    """RSS-style legal safe distance (Maierhofer 2020, ``keeps_safe_distance``)."""
    a_f = abs(r["a_min_ego"])
    a_l = abs(r["a_min_other"])
    return v_follow * r["t_d"] + v_follow ** 2 / (2 * a_f) - v_lead ** 2 / (2 * a_l)


def keeps_safe_distance(ego, leader, r):
    """True iff the longitudinal gap to ``leader`` is at least ``safe_distance``."""
    gap = (leader["x"] - ego["x"]) - r["car_length"]
    return gap >= safe_distance(ego["v"], leader["v"], r)


def safe_gap(ego, others, side, cfg):
    """True iff a lane change to ``ego.lane + side`` is safe (kinematic version).

    Checks the nearest vehicle ahead and behind in the target lane with the
    symmetric safe-distance test. ``side = -1`` is left, ``+1`` is right. In
    Part 3 this is superseded by the velocity-obstacle test in
    ``labs/lab4_velocity_obstacles.py`` (continuous geometry).
    """
    r = cfg["rules"]
    target = ego.get("lane", 0) + side
    if target < 0 or target >= scene_lanes(cfg, ego):
        return False  # no such lane
    lane_vs = [o for o in others if o.get("lane") == target]
    ahead = [o for o in lane_vs if o["x"] >= ego["x"]]
    behind = [o for o in lane_vs if o["x"] < ego["x"]]
    ok = True
    if ahead:
        lead = min(ahead, key=lambda o: o["x"] - ego["x"])
        ok = ok and keeps_safe_distance(ego, lead, r)
    if behind:
        foll = max(behind, key=lambda o: o["x"])
        # the follower must keep a safe distance to the ego after the change
        ok = ok and keeps_safe_distance(foll, ego, r)
    return ok


def scene_lanes(cfg, ego):
    """Number of lanes available (from the env config; falls back generously)."""
    return cfg.get("env", {}).get("config", {}).get("lanes_count", 4)


def _unnecessary_braking(ego, leader, r):
    a = ego.get("a")
    if a is None:
        return False  # acceleration not observable -> cannot assert a violation
    if a >= r["a_abrupt"]:
        return False
    # Abrupt braking is "unnecessary" if there is no leader, or the leader is at a
    # safe distance (i.e. nothing forces the brake).
    return (leader is None) or keeps_safe_distance(ego, leader, r)


def _impedes_flow(ego, leader, r):
    # RG4: ego drives slower than the flow without a slow leader forcing it.
    if leader is not None and (ego["v"] - leader["v"]) > 0:
        return False  # ego is faster than leader -> not the cause
    return ego["v"] < (r["v_max"] - r["delta_v_fl"])


def _passing_on_right(ego, others, r):
    """RI2: ego drives faster than a vehicle to its left, outside exceptions."""
    left_lane = ego.get("lane", 0) - 1
    left_vs = [o for o in others if o.get("lane") == left_lane]
    for o in left_vs:
        if ego["v"] <= o["v"]:
            continue
        # exception: left vehicle is in a queue / slow traffic / congestion and the
        # ego is only slightly faster.
        slow_left = o["v"] <= r["v_qv"]
        slightly = (ego["v"] - o["v"]) <= r["slightly_higher_speed"]
        if slow_left and slightly:
            continue
        return True
    return False


def _makes_uturn(ego, r):
    h = ego.get("heading")
    if h is None:
        return False  # heading not observable -> stubbed (TODO: needs ref-path)
    return abs(h) > r["delta_theta_uturn"]


# =============================================================================
# Stage B — safety shield (FSM over manoeuvres, hard constraints)
# =============================================================================
def safety_shield(action, preds, fsm_state, cfg):
    """Replace an unsafe proposed manoeuvre with the safest legal fallback.

    Args:
        action: int index into ``ACTIONS`` proposed by the policy.
        preds: output of ``predicates``.
        fsm_state: current FSM state name (see labs/lab3_fsm.py).
        cfg: full config.

    Returns:
        ``(safe_action_index, new_fsm_state)``.
    """
    from labs.lab3_fsm import fsm_transition, admissible_actions

    fsm_state = fsm_transition(fsm_state, preds, cfg)
    proposed = ACTIONS[int(action)]
    allowed = admissible_actions(fsm_state, preds, cfg)
    if proposed in allowed:
        return ACTION_INDEX[proposed], fsm_state
    return ACTION_INDEX[safest_fallback(preds, cfg, allowed)], fsm_state


def safest_fallback(preds, cfg, allowed=None):
    """Pick the safest manoeuvre: brake/stop if unsafe, else hold lane."""
    if preds.get("too_close") or preds.get("off_road"):
        return "SLOWER"
    if allowed and "IDLE" in allowed:
        return "IDLE"
    return "SLOWER"


# =============================================================================
# Stage C — logic-shaped reward penalty (soft heuristics)
# =============================================================================
def logic_penalty(preds, cfg):
    """Sum of weighted soft-rule violations: ``Σ λ_i · violation_i`` (>= 0).

    Added (negated) to the reward in ``LogicRewardWrapper`` during the Part-2
    fine-tune. Uses only the *heuristic* rules (RI2, RG4, RG2); hard rules are
    handled by the shield, not the reward.
    """
    lam = cfg["rules"]["lambda"]
    pen = 0.0
    if preds.get("passing_on_right"):
        pen += lam["RI2"]
    if preds.get("impedes_flow"):
        pen += lam["RG4"]
    if preds.get("unnecessary_braking"):
        pen += lam["RG2"]
    return pen


# =============================================================================
# Stage E — independent MTL monitor (audit; NOT what the agent optimises)
# =============================================================================
def rule_violations(preds, cfg):
    """Per-rule boolean violations for one timestep, by an independent monitor.

    Counted separately from the shield/reward so "fewer violations" is not
    circular. The evaluation harness aggregates these into per-rule rates.
    """
    return {
        "RG1": bool(preds.get("too_close")),
        "RG2": bool(preds.get("unnecessary_braking")),
        "RG3": bool(preds.get("over_speed_limit")),
        "RG4": bool(preds.get("impedes_flow")),
        "RI1": bool(preds.get("must_not_stop")),
        "RI2": bool(preds.get("passing_on_right")),
    }
