"""MetaDrive construction with a velocity ``(v, ω)`` action (Part 3).

Public functions:
  * ``make_env_md(cfg, render, seed)`` — a MetaDrive env whose action is a
    continuous, normalised ``(v, ω)`` mapped onto ROS-style ``cmd_vel`` (Lab 1).
  * ``read_scene_md(env)`` — the SAME SI scene schema as ``highway_factory.read_scene``,
    so the NeSy predicates / shield / CBF are reused unchanged.

MetaDrive (0.4.x, gymnasium-native, CPU) is heavy and only installed in the Part-3
notebook, so it is imported lazily inside the functions. Nothing runs at import
time. Warnings are fixed at the source in the notebook setup (legacy ``gym``
removed, ``pygame-ce`` for pkg_resources), not suppressed here.
"""

import numpy as np
import gymnasium as gym


def make_env_md(cfg, render=False, seed=None):
    """Build a MetaDrive env exposing a continuous ``(v, ω)`` action.

    Args:
        cfg: full config; uses the ``metadrive:`` block.
        render: kept for API symmetry with the highway factory — MetaDrive's
            top-down video (``demo/demo_md.py``) renders offscreen regardless, so
            the observation is identical whether or not this is set.
        seed: start scenario seed (MetaDrive uses integer scenario seeds).

    Returns:
        A Gymnasium env whose action is ``(v_norm, ω_norm) ∈ [-1, 1]²``.
    """
    md = cfg["metadrive"]
    try:
        from metadrive.envs import MetaDriveEnv
    except ImportError as e:  # pragma: no cover - MetaDrive is Part-3 only
        raise ImportError(
            "MetaDrive is not installed. In colab_3, install it under the "
            "Python-3.10 (condacolab) runtime: `pip install metadrive-simulator`."
        ) from e

    md_config = {
        "use_render": False,                 # no 3D popup / GPU — CPU state+lidar obs
        "image_observation": False,          # vector observation (MLP-friendly, CPU)
        "traffic_density": md["traffic_density"],
        "num_scenarios": md["num_scenarios"],
        "start_seed": int(seed) if seed is not None else cfg["seed"],
        "horizon": md["horizon"],
        "map": md["map"],
        "vehicle_config": {"lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}},
    }

    env = MetaDriveEnv(md_config)
    env = VelocityActionWrapper(env, cfg)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


class VelocityActionWrapper(gym.Wrapper):
    """Expose a normalised ``(v, ω)`` action and convert it to MetaDrive control.

    The policy (and the manoeuvre→``cmd_vel`` translation in Lab 1) act in velocity
    space; this wrapper turns ``(v_norm, ω_norm) ∈ [-1,1]²`` into MetaDrive's native
    ``(steering, throttle)`` via a proportional velocity controller — the software
    twin of a robot's ``cmd_vel`` low-level controller.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.v_max = cfg["metadrive"]["v_max"]
        self.omega_max = cfg["metadrive"]["omega_max"]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # MetaDrive's BaseEnv.reset() takes only `seed` (no gymnasium `options`), and
        # that seed is a *scenario index* that must lie in
        # ``[start_seed, start_seed + num_scenarios)``. The eval harness passes
        # arbitrary gym-style seeds, so wrap any out-of-range seed into that window.
        if seed is not None:
            try:
                start = int(self.env.config["start_seed"])
                n = max(1, int(self.env.config["num_scenarios"]))
            except Exception:
                start, n = 0, 1
            seed = int(seed)
            if not (start <= seed < start + n):
                seed = start + (seed % n)
        return self.env.reset(seed=seed)

    def step(self, action):
        v_cmd = float(np.clip(action[0], -1.0, 1.0)) * self.v_max
        omega_cmd = float(np.clip(action[1], -1.0, 1.0)) * self.omega_max
        return self.env.step(self._velocity_to_native(v_cmd, omega_cmd))

    def _velocity_to_native(self, v, omega):
        """Map ``(v, ω)`` to MetaDrive's ``(steering, throttle)`` in [-1, 1].

        Proportional controller: steering ∝ ω, throttle ∝ (v_target − v_current).
        MetaDrive has no direct velocity setpoint, so this closes the loop on the
        ego's current speed each step — the same idea as a ``cmd_vel`` PID.
        """
        try:
            v_cur = float(np.linalg.norm(self.env.unwrapped.agent.velocity))
        except Exception:
            v_cur = 0.0
        steering = float(np.clip(omega / max(self.omega_max, 1e-6), -1.0, 1.0))
        throttle = float(np.clip((v - v_cur) / max(self.v_max, 1e-6), -1.0, 1.0))
        return np.array([steering, throttle], dtype=np.float32)


def kin_obs_from_scene(scene, cfg):
    """Reconstruct highway-env's Kinematics observation from an SI scene dict.

    Lets the **discrete Part-2 policy** (trained on highway-env) run on MetaDrive: it
    mirrors ``highway_env.KinematicObservation`` — features ``[presence,x,y,vx,vy]``,
    the ego row absolute, up to ``vehicles_count-1`` nearest neighbours ego-relative and
    distance-sorted, each feature mapped to [-1,1] by highway-env's ranges (MAX_SPEED=40)
    and clipped, then zero-padded to ``vehicles_count`` rows.

    Pure function (scene -> ``(vehicles_count, n_features)`` array) so it is unit-testable
    without MetaDrive. **Approximate**: highway-env's speed scale (~25 m/s) differs from
    MetaDrive's robot scale (~8 m/s), so the ego-speed feature reads lower than the model
    saw in training — the policy still acts, but not identically.
    """
    oc = cfg["env"]["config"]["observation"]
    feats = oc.get("features", ["presence", "x", "y", "vx", "vy"])
    n = oc.get("vehicles_count", 5)
    max_v = 40.0                                   # highway_env Vehicle.MAX_SPEED
    lanes = cfg["env"]["config"].get("lanes_count", 4)
    rng = {"x": (-5 * max_v, 5 * max_v), "y": (-4.0 * lanes, 4.0 * lanes),
           "vx": (-2 * max_v, 2 * max_v), "vy": (-2 * max_v, 2 * max_v)}

    def nz(v, key):
        lo, hi = rng[key]
        return float(np.clip((v - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))

    ego = scene["ego"]

    def make_row(vd, relative):
        bx, by, bvx, bvy = (ego["x"], ego["y"], ego["vx"], ego["vy"]) if relative else (0.0, 0.0, 0.0, 0.0)
        full = {"presence": 1.0,
                "x": nz(vd["x"] - bx, "x"), "y": nz(vd["y"] - by, "y"),
                "vx": nz(vd["vx"] - bvx, "vx"), "vy": nz(vd["vy"] - bvy, "vy")}
        return [full[f] for f in feats]

    rows = [make_row(ego, relative=False)]
    nearest = sorted(scene.get("others", []),
                     key=lambda o: (o["x"] - ego["x"]) ** 2 + (o["y"] - ego["y"]) ** 2)[:n - 1]
    rows += [make_row(o, relative=True) for o in nearest]
    while len(rows) < n:
        rows.append([0.0] * len(feats))
    return np.asarray(rows, dtype=np.float32)


def read_kin_obs_md(env, cfg):
    """Highway-env Kinematics observation reconstructed from a live MetaDrive env."""
    return kin_obs_from_scene(read_scene_md(env), cfg)


def nesy_md_action(part2_model, env, cfg, fsm_state, shield=True):
    """One Lab-1-bridge step: run the Part-2 model on MetaDrive and return its command.

    Reconstruct the highway obs (``read_kin_obs_md``) -> the Part-2 discrete model picks
    a manoeuvre -> [optional FSM shield] -> Lab-1 ``manoeuvre_to_cmd_vel`` -> ``(v, ω)`` ->
    [optional CBF/VO continuous filter] -> normalised MetaDrive action. Shared by the
    Part-3 eval (``eval.evaluate_nesy_md``) and the video demo. Returns ``(action, fsm_state)``.
    """
    from nesy.roadmap import predicates, safety_shield, continuous_shield, ACTIONS
    from labs.lab1_cmd_vel import manoeuvre_to_cmd_vel

    a, _ = part2_model.predict(read_kin_obs_md(env, cfg),
                               deterministic=cfg["eval"].get("deterministic", True))
    scene = read_scene_md(env)
    manoeuvre = ACTIONS[int(a)]
    if shield:                                       # discrete FSM shield (Part 2)
        idx, fsm_state = safety_shield(int(a), predicates(scene, cfg), fsm_state, cfg)
        manoeuvre = ACTIONS[idx]
    v, omega = manoeuvre_to_cmd_vel(manoeuvre, scene, cfg)   # Lab 1: manoeuvre -> (v, ω)
    if shield:                                       # continuous CBF/VO filter (Labs 4/5)
        (v, omega), _ = continuous_shield(v, omega, scene, cfg)
    md = cfg["metadrive"]
    return np.array([v / md["v_max"], omega / md["omega_max"]], dtype=np.float32), fsm_state


def read_scene_md(env):
    """Extract the common SI scene dict from a MetaDrive env.

    Returns the same schema as ``envs.highway_factory.read_scene`` (ego + others in
    metres / m/s, lane as int), so the NeSy predicates / shield / CBF apply unchanged.
    """
    u = env.unwrapped
    ego = getattr(u, "agent", None) or u.vehicle

    def vinfo(obj):
        pos = np.asarray(obj.position, dtype=float)
        vel = np.asarray(obj.velocity, dtype=float)
        try:
            lane = int(obj.lane_index[2])
        except Exception:
            lane = 0
        return {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "v": float(np.linalg.norm(vel)),
            "lane": lane,
            "heading": float(getattr(obj, "heading_theta", 0.0)),
        }

    ego_d = vinfo(ego)
    ego_d["on_road"] = bool(getattr(ego, "on_lane", True))

    others = [vinfo(o) for o in _neighbours(u, ego)]
    return {"ego": ego_d, "others": others, "lanes_count": int(getattr(u, "num_lanes", 3) or 3)}


def _neighbours(u, ego):
    """Traffic vehicles around the ego (best-effort across MetaDrive versions)."""
    try:
        objs = u.engine.get_objects().values()
    except Exception:
        return []
    try:
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        return [o for o in objs if o is not ego and isinstance(o, BaseVehicle)]
    except Exception:
        # Fallback: keep objects that quack like a vehicle (have a lane_index).
        return [o for o in objs if o is not ego and hasattr(o, "lane_index")
                and hasattr(o, "velocity")]


def count_passes_md(env, ahead_ids):
    """Overtake tracking on MetaDrive: vehicles that move from ahead of the ego
    to behind it along the road (+x on the straight ``SSSS`` map) count once.

    Returns ``(passed_this_step, new_ahead_ids)`` — same bookkeeping as the
    highway ``OvertakeCounter``, keyed by MetaDrive object names.
    """
    u = env.unwrapped
    ego = u.agent
    ex = float(ego.position[0])
    passed = 0
    still = set()
    for o in _neighbours(u, ego):
        oid = getattr(o, "name", None) or id(o)
        if oid in ahead_ids:
            if float(o.position[0]) < ex:
                passed += 1
            else:
                still.add(oid)
        elif float(o.position[0]) > ex:
            still.add(oid)
    return passed, still
