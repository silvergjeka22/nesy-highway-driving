"""Lab 4 — MCTS with logical heuristics + velocity obstacles.

Two components from the course:
  * **MCTS** — Monte Carlo Tree Search with logic-guided rollouts.
    Uses a lightweight kinematic forward model (no env cloning) for fast planning.
  * **Velocity obstacles** — continuous "is this gap safe?" test for the (v, ω) action.
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


# MCTS — Monte Carlo Tree Search with logic-guided rollouts
_LANE_WIDTH = 4.0
_CAR_LENGTH = 5.0
_DV = {0: 0.0, 1: 0.0, 2: 0.0, 3: 4.0, 4: -4.0}
_DLANE = {0: -1, 1: 0, 2: 1, 3: 0, 4: 0}


def _sim_scene(scene, action_idx, dt=0.5):
    """Kinematic one-step forward model (no env needed).

    Returns (next_scene, reward, crashed). Actions: 0=lane_left, 1=idle,
    2=lane_right, 3=faster, 4=slower. Vehicles advance at constant velocity.
    """
    ego = scene["ego"]
    new_vx = max(0.0, ego["vx"] + _DV[action_idx])
    new_x = ego["x"] + new_vx * dt
    new_lane = ego["lane"] + _DLANE[action_idx]
    n_lanes = scene["lanes_count"]
    new_lane = max(0, min(n_lanes - 1, new_lane))
    # Real lane-centre y when the scene provides it (MetaDrive), else highway grid.
    centres = ego.get("lane_centres") or []
    new_y = centres[new_lane] if new_lane < len(centres) else new_lane * _LANE_WIDTH

    others_next = []
    for o in scene.get("others", []):
        others_next.append({
            "x": o["x"] + o["vx"] * dt, "y": o["y"],
            "vx": o["vx"], "vy": o.get("vy", 0.0),
            "v": o["v"], "lane": o["lane"],
        })

    new_ego = {
        "x": new_x, "y": new_y,
        "vx": new_vx, "vy": 0.0, "v": new_vx,
        "lane": new_lane, "on_road": 0 <= new_lane < n_lanes,
        "heading": ego.get("heading", 0.0),
    }
    if centres:
        new_ego["lane_centres"] = centres
        new_ego["lanes_count"] = ego.get("lanes_count", n_lanes)
    next_scene = {"ego": new_ego, "others": others_next, "lanes_count": n_lanes}

    crashed = any(
        abs(new_x - o["x"]) < _CAR_LENGTH and o["lane"] == new_lane
        for o in others_next
    )
    reward = -1.0 if crashed else (new_vx / 30.0) * 0.5
    return next_scene, reward, crashed


def mcts_action(env, cfg, scene_fn=None):
    """Pick a discrete action via MCTS with logic-guided rollouts.

    Uses a lightweight kinematic forward model instead of stepping the real env,
    so planning is fast (~1ms per decision vs ~11s with env.step).
    ``scene_fn`` extracts the SI scene dict (default: highway read_scene; pass
    ``read_scene_md`` to plan on MetaDrive).
    """
    from envs.highway_factory import read_scene
    from nesy.roadmap import predicates, ACTIONS
    from labs.lab3_fsm import admissible_actions

    scene_fn = scene_fn or read_scene

    mc = cfg.get("mcts", {})
    n_sims = mc.get("n_simulations", 20)
    depth = mc.get("depth", 3)
    p_h = mc.get("p_heuristic", 0.7)
    gamma = mc.get("gamma", 0.98)
    n_actions = len(ACTIONS)

    scene = scene_fn(env)

    returns = np.zeros(n_actions)
    counts = np.zeros(n_actions)

    preds = predicates(scene, cfg)
    allowed = admissible_actions("CRUISE", preds, cfg)
    allowed_idx = [i for i, name in ACTIONS.items() if name in allowed]

    for _ in range(n_sims):
        if np.random.random() < p_h and allowed_idx:
            root = int(np.random.choice(allowed_idx))
        else:
            root = int(np.random.randint(n_actions))

        sim_scene, reward, crashed = _sim_scene(scene, root)
        total = float(reward)

        for d in range(1, depth):
            if crashed:
                break
            sim_preds = predicates(sim_scene, cfg)
            sim_allowed = admissible_actions("CRUISE", sim_preds, cfg)
            sim_idx = [i for i, name in ACTIONS.items() if name in sim_allowed]

            if np.random.random() < p_h and sim_idx:
                a = int(np.random.choice(sim_idx))
            else:
                a = int(np.random.randint(n_actions))

            sim_scene, reward, crashed = _sim_scene(sim_scene, a)
            total += float(reward) * (gamma ** d)

        returns[root] += total
        counts[root] += 1

    avg = np.full(n_actions, -np.inf)
    mask = counts > 0
    avg[mask] = returns[mask] / counts[mask]
    return int(np.argmax(avg))


class MCTSPolicy:
    """SB3-compatible predict() adapter for MCTS planning (no trained model needed)."""

    def __init__(self, cfg, scene_fn=None):
        self.cfg = cfg
        self.scene_fn = scene_fn
        self._env = None

    def set_eval_env(self, env):
        """Called by evaluate() so MCTS can read scenes from the live env."""
        self._env = env

    def predict(self, obs, deterministic=True):
        if self._env is None:
            raise RuntimeError("MCTSPolicy needs set_eval_env() before predict()")
        return np.array(mcts_action(self._env, self.cfg, self.scene_fn)), None
