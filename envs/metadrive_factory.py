"""MetaDrive env with a continuous (v, ω) action (Part 3).

Lazy imports — MetaDrive is only installed in the Part-3 notebook.
"""

import numpy as np
import gymnasium as gym


def make_env_md(cfg, render=False, seed=None, video_3d=False):
    """Build a MetaDrive env with a normalised (v, ω) action space."""
    md = cfg["metadrive"]
    try:
        from metadrive.envs import MetaDriveEnv
    except ImportError as e:  # pragma: no cover - MetaDrive is Part-3 only
        raise ImportError(
            "MetaDrive is not installed. In colab_3, install it under the "
            "Python-3.10 (condacolab) runtime: `pip install metadrive-simulator`."
        ) from e

    veh_cfg = {"lidar": {"num_lasers": md.get("lidar_num_lasers", 72)}}

    md_config = {
        "use_render": False,
        "image_observation": video_3d,
        "traffic_density": md["traffic_density"],
        "num_scenarios": md["num_scenarios"],
        "start_seed": int(seed) if seed is not None else cfg["seed"],
        "horizon": md["horizon"],
        "map": md["map"],
        "vehicle_config": veh_cfg,
    }

    if video_3d:
        from metadrive.component.sensors.rgb_camera import RGBCamera
        size = tuple(md.get("video_size", [800, 800]))
        veh_cfg["image_source"] = "rgb_camera"
        md_config.update({
            "sensors": {"rgb_camera": (RGBCamera, size[0], size[1])},
            "norm_pixel": False,
            "stack_size": 1,
        })

    env = MetaDriveEnv(md_config)
    env = VelocityActionWrapper(env, cfg)
    if seed is not None:
        env.reset(seed=int(seed))
    return env


class VelocityActionWrapper(gym.Wrapper):
    """Convert normalised (v, ω) to MetaDrive's (steering, throttle) via a P-controller."""

    def __init__(self, env, cfg):
        super().__init__(env)
        self.v_max = cfg["metadrive"]["v_max"]
        self.omega_max = cfg["metadrive"]["omega_max"]
        self.throttle_kp = float(cfg["metadrive"].get("throttle_kp", 5.0))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # Wrap arbitrary gym-style seeds into MetaDrive's [start_seed, start_seed+num_scenarios) window.
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
        """Map (v, ω) to MetaDrive's (steering, throttle) via proportional control."""
        try:
            v_cur = float(np.linalg.norm(self.env.unwrapped.agent.velocity))
        except Exception:
            v_cur = 0.0
        steering = float(np.clip(omega / max(self.omega_max, 1e-6), -1.0, 1.0))
        throttle = float(np.clip(self.throttle_kp * (v - v_cur) / max(self.v_max, 1e-6), -1.0, 1.0))
        return np.array([steering, throttle], dtype=np.float32)


def kin_obs_from_scene(scene, cfg):
    """Reconstruct highway-env's Kinematics observation from an SI scene dict.

    Lets the Part-2 discrete policy run on MetaDrive by mirroring the highway-env
    observation format. obs_speed_scale corrects the MetaDrive->highway speed gap.
    """
    oc = cfg["env"]["config"]["observation"]
    feats = oc.get("features", ["presence", "x", "y", "vx", "vy"])
    n = oc.get("vehicles_count", 5)
    max_v = 40.0
    lanes = cfg["env"]["config"].get("lanes_count", 4)
    vscale = float(cfg.get("metadrive", {}).get("obs_speed_scale", 1.0))
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
                "vx": nz((vd["vx"] - bvx) * vscale, "vx"), "vy": nz((vd["vy"] - bvy) * vscale, "vy")}
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


def scene_to_lidar(scene, n_lasers=72, max_range=50.0):
    """Convert scene vehicle positions into pseudo-lidar ranges for Lab 2 integration."""
    ego = scene["ego"]
    ranges = np.full(n_lasers, max_range)
    for o in scene.get("others", []):
        dx = o["x"] - ego["x"]
        dy = o["y"] - ego["y"]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        angle = np.arctan2(dy, dx)
        idx = int((angle + np.pi) / (2 * np.pi) * n_lasers) % n_lasers
        ranges[idx] = min(ranges[idx], dist)
    return ranges


def nesy_md_action(part2_model, env, cfg, fsm_state, shield=True):
    """Run the Part-2 discrete model on MetaDrive via the Lab-1 bridge. Returns (action, fsm_state)."""
    from nesy.roadmap import predicates, safety_shield, continuous_shield, ACTIONS
    from labs.lab1_cmd_vel import manoeuvre_to_cmd_vel

    a, _ = part2_model.predict(read_kin_obs_md(env, cfg),
                               deterministic=cfg["eval"].get("deterministic", True))
    scene = read_scene_md(env)
    manoeuvre = ACTIONS[int(a)]
    if shield:
        idx, fsm_state = safety_shield(int(a), predicates(scene, cfg), fsm_state, cfg)
        manoeuvre = ACTIONS[idx]
    v, omega = manoeuvre_to_cmd_vel(manoeuvre, scene, cfg)
    if shield:
        lidar = scene_to_lidar(scene, cfg["metadrive"].get("lidar_num_lasers", 72))
        (v, omega), _ = continuous_shield(v, omega, scene, cfg, lidar_ranges=lidar)
    md = cfg["metadrive"]
    return np.array([v / md["v_max"], omega / md["omega_max"]], dtype=np.float32), fsm_state


def read_scene_md(env):
    """Extract the SI scene dict from a MetaDrive env (same schema as highway read_scene)."""
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
        return [o for o in objs if o is not ego and hasattr(o, "lane_index")
                and hasattr(o, "velocity")]


def count_passes_md(env, ahead_ids):
    """Count vehicles that moved from ahead to behind the ego. Returns (passed, new_ahead_ids)."""
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
