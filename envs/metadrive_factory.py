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


def filter_action_md(action, scene, cfg):
    """Apply the continuous CBF/VO shield to a normalised ``(v_norm, ω_norm)`` action.

    Decodes the action to SI ``(v, ω)``, runs ``nesy.roadmap.continuous_shield``
    (CBF + velocity obstacles — the same hard rules as the discrete shield), and
    re-encodes to the normalised action the env expects. Used as ``evaluate``'s
    ``action_filter`` and by ``demo/demo_md.py``, so the "+CBF/VO" config enforces
    the hard rules on the continuous command.
    """
    from nesy.roadmap import continuous_shield

    md = cfg["metadrive"]
    v_cmd = float(action[0]) * md["v_max"]
    omega_cmd = float(action[1]) * md["omega_max"]
    (v, omega), _ = continuous_shield(v_cmd, omega_cmd, scene, cfg)
    return np.array([v / md["v_max"], omega / md["omega_max"]], dtype=np.float32)


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
