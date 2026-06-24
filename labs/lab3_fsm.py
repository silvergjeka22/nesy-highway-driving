"""Lab 3 — finite-state-machine behaviour layer (the shield host).

The FSM over high-level driving states is where the temporal-logic rules and the
shield's fallbacks live. ``admissible_actions`` encodes the **hard constraints**
(RG1 safe gap, RG3 speed limit, RI1 no-stop, stay-on-road) as a per-state mask
over the discrete meta-actions; ``fsm_transition`` moves between states from the
predicate truth values.

Function-only. The discrete-action vocabulary matches highway-env's
``DiscreteMetaAction`` (see ``nesy/roadmap.ACTIONS``).
"""

STATES = ["CRUISE", "FOLLOW", "OVERTAKE_LEFT", "MERGE", "EMERGENCY_STOP"]
# Part-3 (MetaDrive) intersection states, from the 2022 paper. The transitions
# into these need the intersection observation (stop sign / light / priority),
# which is grounded in envs/metadrive_factory.py -> nesy.intersection_predicates.
INTERSECTION_STATES = ["STOP_SIGN_WAIT", "YIELD", "LIGHT_STOP"]
ALL_ACTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def fsm_transition(state, preds, cfg):
    """Next FSM state from the current state and predicates.

    CRUISE  -> EMERGENCY_STOP if too_close/off_road
            -> FOLLOW         if a leader is within follow_gap
    FOLLOW  -> OVERTAKE_LEFT  if the left gap is safe (want to pass)
            -> CRUISE         if no near leader
    OVERTAKE_LEFT -> CRUISE   once clear
    any     -> EMERGENCY_STOP on imminent danger; recover to CRUISE when safe.
    """
    f = cfg["fsm"]
    gap = preds.get("leader_gap", float("inf"))

    if preds.get("too_close") or preds.get("off_road") or gap < f["emergency_gap"]:
        return "EMERGENCY_STOP"

    if state == "EMERGENCY_STOP":
        return "CRUISE" if gap >= f["follow_gap"] else "FOLLOW"

    if state in ("CRUISE", "FOLLOW"):
        if gap < f["follow_gap"]:
            if preds.get("safe_gap_left"):
                return "OVERTAKE_LEFT"
            return "FOLLOW"
        return "CRUISE"

    if state == "OVERTAKE_LEFT":
        return "CRUISE" if gap >= f["follow_gap"] else "OVERTAKE_LEFT"

    return "CRUISE"


def admissible_actions(state, preds, cfg):
    """Hard-constraint mask: the manoeuvres allowed in ``state`` given ``preds``.

    This is where the safety rules bite:
      * RG1  — forbid a lane change into an unsafe gap.
      * RG1  — forbid FASTER when the leader gap is unsafe.
      * RG3  — forbid FASTER above the speed limit.
      * stay-on-road / RI1 — in EMERGENCY_STOP only SLOWER/IDLE are allowed.
    """
    if state == "EMERGENCY_STOP":
        return ["SLOWER", "IDLE"]

    # Part-3 intersection states: must be able to come to a stop / yield.
    # TODO: refine per-state masks once MetaDrive intersection predicates are wired
    # (must_stop_at_line, has_right_of_way, light_is_red).
    if state in INTERSECTION_STATES:
        return ["SLOWER", "IDLE"]

    allowed = list(ALL_ACTIONS)

    # RG1 + RG3: gate acceleration.
    if preds.get("too_close") or preds.get("over_speed_limit"):
        _drop(allowed, "FASTER")

    # RG1: gate lane changes on the target-lane safe gap.
    if not preds.get("safe_gap_left"):
        _drop(allowed, "LANE_LEFT")
    if not preds.get("safe_gap_right"):
        _drop(allowed, "LANE_RIGHT")

    # State preference: while overtaking, do not voluntarily merge right.
    if state == "OVERTAKE_LEFT":
        _drop(allowed, "LANE_RIGHT")

    return allowed or ["SLOWER"]


def _drop(lst, item):
    if item in lst:
        lst.remove(item)
